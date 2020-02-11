import torch
import random

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim,
                 enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout_rate = dropout

        self.embedding = torch.nn.Embedding(input_dim, emb_dim)
        self.rnn = torch.nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = torch.nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(
            self.fc(
                torch.cat(
                    (hidden[-2, :, :], hidden[-1, :, :]),
                    dim=1
                )
            )
        )
        return outputs, hidden


class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = torch.nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(
            self.attn(
                torch.cat(
                    (repeated_decoder_hidden, encoder_outputs),
                    dim=2)
            )
        )
        attention = torch.sum(energy, dim=2)

        return torch.nn.functional.softmax(attention, dim=1)


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, emb_dim,
                 enc_hid_dim, dec_hid_dim,
                 dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.attention = attention

        self.embedding = torch.nn.Embedding(output_dim, emb_dim)

        self.rnn = torch.nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = torch.nn.Linear(self.attention.attn_in + emb_dim,
                                   output_dim)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def _weighted_encoder_rep(self,
                              decoder_hidden,
                              encoder_outputs):

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self,
                input,
                decoder_hidden,
                encoder_outputs):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(torch.nn.Module):
    def __init__(self,
                 encoder,
                 decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, target_bos_idx, trg=None,
                teacher_forcing_ratio: float = 0.5, max_len=20):
        batch_size = src.shape[1]
        max_len = max_len if trg is None else trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = torch.tensor([target_bos_idx] * src.shape[1])

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (top1 if not teacher_force else trg[t])

        return outputs
