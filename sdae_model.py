# -*- coding: utf-8 -*-
"""
@refer: https://github.com/jihunchoi/sequential-denoising-autoencoder-tf
@usage: sequential-denoising-autoencoder for sentence embeddding
"""
"""Model configuration object."""
from configparser import ConfigParser
import tensorflow as tf
from tensorflow.contrib import seq2seq,layers, rnn, slim
from tensorflow.contrib.framework import get_or_create_global_step
import argparse
import os
from datetime import datetime
import random
import numpy as np

logging = tf.logging
logging.set_verbosity(logging.INFO)

class DataGenerator(object):

    """A data generator class."""

    def pad_batch(self, data_batch, prepend_eos, append_eos):
        max_len = max(len(d) for d in data_batch)
        prefix = []
        suffix = []
        if prepend_eos:
            prefix = [self.word_to_id(self.eos_symbol)]
        if append_eos:
            suffix = [self.word_to_id(self.eos_symbol)]
        return [prefix + d + suffix + [0]*(max_len - len(d))
                for d in data_batch]

    def add_noise(self, word_ids):
        word_ids = word_ids.copy()
        # First, omit some words
        num_omissions = int(self.omit_prob * len(word_ids))
        inds_to_omit = np.random.permutation(len(word_ids))[:num_omissions]
        for i in inds_to_omit:
            word_ids[i] = self.word_to_id(self.unk_symbol)
        # Second, swap some of adjacent words
        num_swaps = int(self.swap_prob * (len(word_ids) - 1))
        inds_to_swap = np.random.permutation(len(word_ids) - 1)[:num_swaps]
        for i in inds_to_swap:
            word_ids[i], word_ids[i+1] = word_ids[i+1], word_ids[i]
        return word_ids

    def word_to_id(self, word):
        if word not in self.vocab:
            return self.vocab[self.unk_symbol]
        return self.vocab[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word=w) for w in words]

    def id_to_word(self, id_):
        return self.reverse_vocab[id_]

    def ids_to_words(self, ids):
        return [self.id_to_word(id_=id_) for id_ in ids]

    def __init__(self, data_path, vocab_path, eos_symbol, unk_symbol,
                 omit_prob, swap_prob, batch_size, max_length, max_epoch):
        self.eos_symbol = eos_symbol
        self.unk_symbol = unk_symbol
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.omit_prob = omit_prob
        self.swap_prob = swap_prob

        self.data = [d for d in self.read_data(data_path)
                     if len(d) < max_length]
        self.vocab = self.read_vocab(vocab_path)
        self.reverse_vocab = dict((i, w) for w, i in self.vocab.items())

        self._epoch = 0
        self._progress_in_epoch = 0

    @property
    def progress(self):
        return self._epoch + self._progress_in_epoch

    def construct_data(self, words_batch):
        word_ids_batch = [self.words_to_ids(words=words)
                          for words in words_batch]
        length = np.array([len(d) for d in word_ids_batch],
                          dtype=np.int32)
        noise_word_ids_batch = [self.add_noise(word_ids)
                                for word_ids in word_ids_batch]
        inputs = np.array(
            self.pad_batch(noise_word_ids_batch,
                           prepend_eos=False, append_eos=True),
            dtype=np.int32)
        targets = np.array(
            self.pad_batch(word_ids_batch,
                           prepend_eos=True, append_eos=True),
            dtype=np.int32)
        inputs_length = length + 1
        targets_length = length + 2
        return inputs, inputs_length, targets, targets_length

    def __iter__(self):
        for self._epoch in range(self.max_epoch):
            random.shuffle(self.data)
            for i in range(0, len(self.data), self.batch_size):
                inputs, inputs_length, targets, targets_length = (
                    self.construct_data(self.data[i : i+self.batch_size]))
                yield inputs, inputs_length, targets, targets_length
                self._progress_in_epoch = i / len(self.data)

    def sample(self, num_samples):
        sample_inds = np.random.permutation(len(self.data))[:num_samples]
        words_sample = [self.data[i] for i in sample_inds]
        inputs, inputs_length, targets, targets_length = (
            self.construct_data(words_sample))
        return inputs, inputs_length, targets, targets_length
    @staticmethod
    def read_data(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for sentence in f:
                words = sentence.split()
                data.append(words)
        return data
    @staticmethod
    def read_vocab(vocab_path):
        vocab = dict()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                w, i = line.split()
                vocab[w] = int(i)
        return vocab


"""Sequential autoencoder implementation."""
class Model(object):
    def __init__(self, config, data_generator):
        self.config = config
        self.data_generator = data_generator

    def build_model(self):
        config = self.config
        data_generator = self.data_generator
        logging.info('Building the model...')
        # Placeholders
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inputs')
        self.inputs_length = tf.placeholder(dtype=tf.int32, shape=[None], name='inputs_length')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
        self.targets_length = tf.placeholder(dtype=tf.int32, shape=[None], name='targets_length')

        vocab_size = len(data_generator.vocab)
        embeddings = tf.get_variable(name='embeddings', shape=[vocab_size, config.word_dim], dtype=tf.float32)

        with tf.variable_scope('decoder'):
            with tf.variable_scope('output') as output_scope:
                # This variable-scope-trick is used to ensure that
                # output_fn has a proper scope regardless of a caller's
                # scope.
                def output_fn(cell_outputs):
                    return layers.fully_connected(inputs=cell_outputs, num_outputs=vocab_size, activation_fn=None,
                        scope=output_scope)

        self.rnn_cell = rnn.GRUBlockCell(config.sentence_dim)
        self.encoder_state = self.encode(cell=self.rnn_cell, embeddings=embeddings, inputs=inputs, inputs_length=inputs_length,
            scope='encoder')
        self.decoder_outputs = self.decode_train(cell=self.rnn_cell, embeddings=embeddings, encoder_state=self.encoder_state,
            targets=self.targets[:, :-1], targets_length=self.targets_length - 1, scope='decoder')
        self.generated = self.decode_inference(cell=self.rnn_cell, embeddings=embeddings, encoder_state=self.encoder_state,
            output_fn=output_fn, vocab_size=vocab_size, bos_id=data_generator.vocab['<EOS>'],
            eos_id=data_generator.vocab['<EOS>'], max_length=config.max_length, scope='decoder', reuse=True)
        self.loss = self.loss(decoder_outputs=self.decoder_outputs, output_fn=output_fn, targets=targets[:, 1:],
                        targets_length=self.targets_length - 1)

        self.global_step = get_or_create_global_step()
        self.train_op = slim.optimize_loss(loss=self.loss, global_step=self.global_step, learning_rate=None,
            optimizer=tf.train.AdamOptimizer(), clip_gradients=5.0)

        self.summary_writer = tf.summary.FileWriter(logdir=os.path.join(config.save_dir, 'log'))
        self.summary = tf.summary.merge_all()

        tf.get_variable_scope().set_initializer(tf.random_normal_initializer(mean=0.0, stddev=0.01))
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=20)

    @staticmethod
    def encode(cell, embeddings, inputs, inputs_length, scope='encoder', reuse=None):
        """
        Args:
            cell: An RNNCell object
            embeddings: An embedding matrix with shape
                (vocab_size, word_dim) and with float32 type
            inputs: A int32 tensor with shape (batch, max_len), which
                contains word indices
            inputs_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch
            scope: A VariableScope object of a string which indicates
                the scope
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            sent_vec, which is a int32 tensor with shape
            (batch, cell.output_size) that contains sentence representations
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer(), reuse=reuse):
            # inputs_embed: (batch, max_len, word_dim)
            inputs_embed = tf.nn.embedding_lookup(params=embeddings, ids=inputs)
            _, sent_vec = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_embed, sequence_length=inputs_length,
                dtype=tf.float32, time_major=False, scope='rnn')
        return sent_vec

    @staticmethod
    def decode_train(cell, embeddings, encoder_state, targets, targets_length, scope='decoder', reuse=None):
        """
        Args:
            cell: An RNNCell object
            embeddings: An embedding matrix with shape
                (vocab_size, word_dim)
            encoder_state: A tensor that contains the encoder state;
                its shape should match that of cell.zero_state
            targets: A int32 tensor with shape (batch, max_len), which
                contains word indices; should start and end with
                the proper <BOS> and <EOS> symbol
            targets_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch
            scope: A VariableScope object of a string which indicates
                the scope
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            decoder_outputs, which is a float32
            (batch, max_len, cell.output_size) tensor that contains
            the cell's hidden state per time step
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer(), reuse=reuse):
            decoder_fn = seq2seq.simple_decoder_fn_train(encoder_state=encoder_state)
            targets_embed = tf.nn.embedding_lookup(params=embeddings, ids=targets)
            decoder_outputs, _, _ = seq2seq.dynamic_rnn_decoder(cell=cell, decoder_fn=decoder_fn, inputs=targets_embed,
                sequence_length=targets_length, time_major=False, scope='rnn')
        return decoder_outputs
    @staticmethod
    def loss(decoder_outputs, output_fn, targets, targets_length):
        """
        Args:
            decoder_outputs: A return value of decode_train function
            output_fn: A function that projects a vector with length
                cell.output_size into a vector with length vocab_size
            targets: A int32 tensor with shape (batch, max_len), which
                contains word indices
            targets_length: A int32 tensor with shape (batch,), which
                contains the length of each sample in a batch

        Returns:
            loss, which is a scalar float32 tensor containing an average
            cross-entropy loss value
        """

        logits = output_fn(decoder_outputs)
        max_len = logits.get_shape()[1].value
        if max_len is None:
            max_len = tf.shape(logits)[1]
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        losses_mask = tf.sequence_mask(lengths=targets_length, maxlen=max_len, dtype=tf.float32)
        return tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)
    @staticmethod
    def decode_inference(cell, embeddings, encoder_state, output_fn, vocab_size, bos_id, eos_id, max_length,
                         scope='decoder', reuse=None):
        """
        Args:
            cell: An RNNCell object
            embeddings: An embedding matrix with shape
                (vocab_size, word_dim)
            encoder_state: A tensor that contains the encoder state;
                its shape should match that of cell.zero_state
            output_fn: A function that projects a vector with length
                cell.output_size into a vector with length vocab_size;
                please beware of the scope, since it will be called inside
                'scope/rnn' scope
            vocab_size: The size of a vocabulary set
            bos_id: The ID of the beginning-of-sentence symbol
            eos_id: The ID of the end-of-sentence symbol
            max_length: The maximum length of a generated sentence;
                it stops generating words when this number of words are
                generated and <EOS> is not appeared till then
            scope: A VariableScope object of a string which indicates
                the scope
            reuse: A boolean value or None which specifies whether to
                reuse variables already defined in the scope

        Returns:
            generated, which is a float32 (batch, <=max_len)
            tensor that contains IDs of generated words
        """

        with tf.variable_scope(scope, initializer=tf.orthogonal_initializer(), reuse=reuse):
            decoder_fn = seq2seq.simple_decoder_fn_inference(output_fn=output_fn, encoder_state=encoder_state,
                embeddings=embeddings, start_of_sequence_id=bos_id, end_of_sequence_id=eos_id,
                maximum_length=max_length, num_decoder_symbols=vocab_size)
            generated_logits, _, _ = seq2seq.dynamic_rnn_decoder(cell=cell, decoder_fn=decoder_fn, time_major=False,
                scope='rnn')
        generated = tf.argmax(generated_logits, axis=2)
        return generated

    def train(self):
        config = self.config
        data_generator = self.data_generator
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        with tf.Graph().as_default() as graph:
            with tf.Session() as sess:
                logging.info('Training starts!')
                for data_batch in data_generator:
                    (inputs_v, inputs_length_v, targets_v, targets_length_v) = data_batch
                    summary_v, global_step_v, _ = sess.run(fetches=[self.summary, self.global_step, self.train_op],
                        feed_dict={inputs: inputs_v,
                                   inputs_length: inputs_length_v,
                                   targets: targets_v,
                                   targets_length: targets_length_v})
                    self.summary_writer.add_summary(summary=summary_v, global_step=global_step_v)
                    if global_step_v % 100 == 0:
                        logging.info(
                            '{} Iter #{}, Epoch {:.2f}'.format(datetime.now(), global_step_v, data_generator.progress))
                        num_samples = 2
                        (inputs_sample_v, inputs_length_sample_v, targets_sample_v, targets_length_sample_v) = (
                            data_generator.sample(num_samples))
                        generated_v = sess.run(fetches=self.generated,
                            feed_dict={inputs: inputs_sample_v, inputs_length: inputs_length_sample_v})
                        for i in range(num_samples):
                            logging.info('-' * 60)
                            logging.info('Sample #{}'.format(i))
                            inputs_sample_words = data_generator.ids_to_words(
                                inputs_sample_v[i][:inputs_length_sample_v[i]])
                            targets_sample_words = data_generator.ids_to_words(
                                targets_sample_v[i][1:targets_length_sample_v[i]])
                            generated_words = data_generator.ids_to_words(generated_v[i])
                            if '<EOS>' in generated_words:
                                eos_index = generated_words.index('<EOS>')
                                generated_words = generated_words[:eos_index + 1]
                            logging.info('Input: {}'.format(' '.join(inputs_sample_words)))
                            logging.info('Target: {}'.format(' '.join(targets_sample_words)))
                            logging.info('Generated: {}'.format(' '.join(generated_words)))
                        logging.info('-' * 60)

                    if global_step_v % 500 == 0:
                        save_path = os.path.join(config.save_dir, 'model.ckpt')
                        real_save_path = self.saver.save(sess=sess, save_path=save_path, global_step=global_step_v)
                        logging.info('Saved the checkpoint to: {}'.format(real_save_path))


def add_parser():
    parser = argparse.ArgumentParser(description='Train the SDAE sae.')
    parser.add_argument('--data', required=True,
                        help='The path of a data file')
    parser.add_argument('--vocab', required=True,
                        help='The path of a vocabulary file')
    parser.add_argument('--save-dir', required=True,
                        help='The path to save sae files')
    parser.add_argument('--word-dim', type=int, default=None,
                        help='The dimension of a word representation')
    parser.add_argument('--sentence-dim', type=int, default=None,
                        help='The dimension of a sentence representation')
    parser.add_argument('--omit-prob', type=float, default=None,
                        help='A probability of a word to be omitted')
    parser.add_argument('--swap-prob', type=float, default=None,
                        help='A probability of adjacent two words to be '
                             'swapped')
    parser.add_argument('--config', default=None,
                        help='The path of a model configuration file')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='The size of a mini-batch')
    parser.add_argument('--max-epoch', type=int, default=5,
                        help='The maximum epoch number')
    parser.add_argument('--max-length', type=int, default=50,
                        help='The maximum number of words; sentences '
                             'longer than this number are ignored')
    args = parser.parse_args()
    return args
def main():
    config = add_parser()

    logging.info('Initializing the data generator...')
    data_generator = DataGenerator(
        data_path=config.data_path, vocab_path=config.vocab_path,
        eos_symbol='<EOS>', unk_symbol='<UNK>',
        omit_prob=config.omit_prob, swap_prob=config.swap_prob,
        batch_size=config.batch_size, max_length=config.max_length, max_epoch=config.max_epoch)
    logging.info('Start training SDAE model...')
    model = Model(config, data_generator)
    model.train()
    logging.info('Done')



if __name__ == '__main__':

    main()
