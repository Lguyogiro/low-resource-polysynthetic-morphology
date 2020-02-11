import argparse
import math
import random
import torch
from models import Encoder, Decoder, Seq2Seq, Attention
from torchtext.data import Field, BucketIterator, Example, Dataset


class Seq2SeqMorphoSegmenter(object):
    """
    Implementation of a sequence-to-sequence model for
    morphological segmentation of low-resource_languages as described in
    https://www.aclweb.org/anthology/N18-1005.pdf. The default parameters
    reproduce the baseline experiment.

    TODO: Support multi-task training and data augmentation experiments.
    """
    def __init__(self, encoder_embedding_dim=300, encoder_hidden_size=100,
                 encoder_dropout_p=0.3, decoder_embedding_dim=300,
                 decoder_hidden_size=100, decoder_dropout_p=0.3,
                 attention_length=15, learning_rate=0.001, gradient_clip=1,
                 batch_size=20):

        self.enc_emb_dim = encoder_embedding_dim
        self.enc_hidden_size = encoder_hidden_size
        self.enc_dropout = encoder_dropout_p
        self.dec_emb_dim = decoder_embedding_dim
        self.dec_hidden_size = decoder_hidden_size
        self.decoder_dropout = decoder_dropout_p
        self.attn_length = attention_length

        self.source_field = Field(lower=True,
                                  tokenize=lambda s: s.split())

        self.target_field = Field(lower=True,
                                  tokenize=lambda s: s.split())
        self.input_dim = None
        self.output_dim = None
        self.encoder = None
        self.attention = None
        self.decoder = None
        self.model = None
        self.batch_size = batch_size
        self.lr = learning_rate
        self.optimizer = None
        self.clip = gradient_clip

    def _init_weights(self):
        if self.model is None:
            raise ValueError("Model not yet defined, so we can't initialize "
                             "the weights.")

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                torch.nn.init.constant_(param.data, 0)

    def train_single_epoch(self, iterator, criterion):
        self.model.train()
        epoch_loss = 0

        for _, batch in enumerate(iterator):
            src = batch.source
            trg = batch.target

            self.optimizer.zero_grad()
            output = self.model(src, self.target_field.vocab.stoi['<sos>'],
                                trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output,
                             trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clip)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate(self, iterator, criterion):
        self.model.eval()
        epoch_loss = 0
        num_attempts = 0
        num_correct = 0
        with torch.no_grad():
            for batch_count, batch in enumerate(iterator):
                src = batch.source
                trg = batch.target

                #
                # Turn off teacher forcing, make prediction, and compare to
                # target.
                #
                output = self.model(src, self.target_field.vocab.stoi['<sos>'],
                                    trg, teacher_forcing_ratio=0)
                for i, item in enumerate(output.transpose(1, 0)):
                    seq = item.argmax(1)
                    target = trg.transpose(1, 0)[i]
                    for j in range(1, len(target)):
                        if target[j] == self.target_field.vocab.stoi['<eos>']:
                            num_correct += 1
                            break
                        elif target[j] != seq[j]:
                            break
                    num_attempts += 1

                loss = criterion(output[1:].view(-1, output.shape[-1]),
                                 trg[1:].view(-1))
                epoch_loss += loss.item()
        return epoch_loss / len(iterator), num_correct / num_attempts

    def predict_one(self, source_string):
        tokenized = self.source_field.preprocess(source_string)
        source_tensor = self.source_field.process([tokenized])

        output = self.model(source_tensor,
                            self.target_field.vocab.stoi['<sos>'],
                            teacher_forcing_ratio=0)
        prediction = []
        for idx in output.transpose(1, 0).squeeze().argmax(dim=1)[1:]:
            if idx == self.target_field.vocab.stoi['<eos>']:
                break
            else:
                prediction.append(self.target_field.vocab.itos[idx])

        return ' '.join(prediction)

    def predict(self, source_strings):
        preprocessed = [self.source_field.preprocess(source_string)
                        for source_string in source_strings]
        source_tensor = self.source_field.process(preprocessed)
        output = self.model(source_tensor,
                            self.target_field.vocab.stoi['<sos>'],
                            teacher_forcing_ratio=0)
        predictions = []
        for example in output.transpose(1, 0).argmax(dim=2):
            pred = []
            for idx in example[1:]:
                if idx == self.target_field.vocab.stoi['<eos>']:
                    break
                else:
                    pred.append(self.target_field.vocab.itos[idx])
            predictions.append(' '.join(pred))

        return predictions

    def fit_and_eval(self, train_source, train_target,
                     eval_source, eval_target, num_epochs=100):
        fields = [('source', self.source_field), ('target', self.target_field)]
        train_examples = [
            Example.fromlist([train_source[i], train_target[i]], fields)
            for i in range(len(train_source))
        ]
        train_dataset = Dataset(train_examples, fields)

        self.source_field.build_vocab(train_dataset)
        self.target_field.build_vocab(train_dataset)

        eval_examples = [
            Example.fromlist([eval_source[i], eval_target[i]], fields)
            for i in range(len(eval_source))
        ]
        val_dataset = Dataset(eval_examples, fields)

        self.input_dim = len(self.source_field.vocab)
        self.output_dim = len(self.target_field.vocab)

        self.encoder = Encoder(self.input_dim,
                               self.enc_emb_dim,
                               self.enc_hidden_size,
                               self.dec_hidden_size,
                               self.enc_dropout)

        self.attention = Attention(self.enc_hidden_size,
                                   self.dec_hidden_size,
                                   self.attn_length)

        self.decoder = Decoder(self.output_dim,
                               self.dec_emb_dim,
                               self.enc_hidden_size,
                               self.dec_hidden_size,
                               self.decoder_dropout,
                               self.attention)

        self.model = Seq2Seq(self.encoder, self.decoder)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.target_field.vocab.stoi['<pad>']
        )
        train_iterator = BucketIterator(train_dataset, batch_size=self.batch_size)
        val_iterator = BucketIterator(val_dataset, batch_size=self.batch_size)

        train_losses, validation_losses, validation_accuracies = [], [], []
        for epoch in range(num_epochs):
            train_loss = self.train_single_epoch(train_iterator, criterion)
            valid_loss, val_acc = self.evaluate(val_iterator, criterion)
            train_losses.append(train_loss)
            validation_losses.append(valid_loss)
            validation_accuracies.append(val_acc)

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tTrain Loss: {train_loss:.3f} '
                  f'| Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} '
                  f'|  Val. PPL: {math.exp(valid_loss):7.3f}')
            print('Val Acc: {}'.format(val_acc))

        return {"training_loss": train_losses,
                "validation_loss": validation_losses,
                "validation_accuracy": validation_accuracies}


def get_data(path_to_source, path_to_target, task=None,
             random_proportion=0.0, path_to_unlabeled_data=None):
    """

    :param path_to_source:
    :param path_to_target:
    :param task:
    :param random_proportion:
    :param path_to_unlabeled_data:
    :return:
    """
    with open(path_to_source) as f:
        source_words = ["<sos> {} <eos>".format(line.strip('\n'))
                        for line in f]
    with open(path_to_target) as f:
        target_words = ["<sos> {} <eos>".format(line.strip('\n'))
                        for line in f]

    task_token = ""
    if task is not None:
        if task == 'multi':
            if random_proportion > 0:
                task_token = 'MTT=r'
            elif path_to_unlabeled_data is not None:
                task_token = 'MTT=u'
        elif task == 'augment':
            task_token = '<sos>'
        else:
            raise ValueError('Unknown task "{}"'.format(task))

        if path_to_unlabeled_data is not None:  # task is MTT=u
            with open(path_to_unlabeled_data) as uf:
                for w in uf:
                    source_word = "{} {} <eos>".format(task_token,
                                                       w.strip('\n'))
                    target_word = "<sos> {} <eos>".format(w.strip('\n'))
                    source_words.append(source_word)
                    target_words.append(target_word)

        elif random_proportion > 0:
            vocab_for_gen = list({ch for word in source_words
                                  for ch in word.split()
                                  if ch not in ('<eos>', '<sos>')})
            num_samples_to_add = int(
                len(source_words) * args.random_proportion
            )
            train_word_lengths = {len(word.split()) for word in
                                  source_words}
            shortest = min(train_word_lengths) - 2  # ignore sos and eos
            longest = max(train_word_lengths) - 2  # ignore sos and eos

            for i in range(num_samples_to_add):
                new_w = []
                new_word_length = random.randint(shortest, longest)
                for new_ch_idx in range(new_word_length):
                    new_w.append(random.choice(vocab_for_gen))
                source_words.append("{} {} <eos>".format(task_token,
                                                         ' '.join(new_w)))
                target_words.append("<sos> {} <eos>".format(' '.join(new_w)))

    return source_words, target_words


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('train_source_words_file')
    argparser.add_argument('train_target_words_file')
    argparser.add_argument('eval_source_words_file')
    argparser.add_argument('eval_target_words_file')
    argparser.add_argument('--random_proportion', '-r',
                           type=float,
                           default=0,
                           help="Proportion of the original training data "
                                "size that will be random strings with which "
                                "to augment the training data. eg -r 1 means "
                                "'increase the training data by 100%% using "
                                "random strings'.")
    argparser.add_argument('--unlabeled_words_file', '-u',
                           help="Path to a file containing words from some "
                                "corpus, without segmentation labels. These "
                                "will either be used to augment the training "
                                "set or for multi-task training.")
    argparser.add_argument('--multitask', '-m', action='store_true')

    args = argparser.parse_args()
    if args.multitask:
        task = 'multi'
    else:
        task = 'augment'

    source_words_train, target_words_train = get_data(
        args.train_source_words_file, args.train_target_words_file,
        task=task, random_proportion=args.random_proportion,
        path_to_unlabeled_data=args.unlabeled_words_file
    )
    source_words_val, target_words_val = get_data(args.eval_source_words_file,
                                                  args.eval_target_words_file)
    segmenter = Seq2SeqMorphoSegmenter()
    segmenter.fit_and_eval(source_words_train, target_words_train,
                           source_words_val, target_words_val)