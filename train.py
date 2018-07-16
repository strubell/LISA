import tensorflow as tf
import argparse
import dataset
import vocab
import os
from LISA_model import LISAModel
import numpy as np
import data_generator

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_file', type=str, help='Training data file')
arg_parser.add_argument('--save_dir', type=str, help='Training data file')
arg_parser.add_argument('--word_embedding_file', type=str, help='File containing pre-trained word embeddings')

args = arg_parser.parse_args()

data_config = {
      'id': {
        'idx': 0,
      },
      'word': {
        'idx': 3,
        'feature': True,
        'vocab': 'glove.6B.100d.txt',
        'converter': 'lowercase',
        'oov': True
      },
      'auto_pos': {
        'idx': 4,
        'vocab': 'gold_pos'
      },
      'gold_pos': {
        'idx': 5,
        'label': True,
        'vocab': 'gold_pos'
      },
      'parse_head': {
        'idx': 6,
        'label': True,
        'converter': 'parse_roots_self_loop'
      },
      'parse_label': {
        'idx': 7,
        'label': True,
        'vocab': 'parse_label'
      },
      'domain': {
        'idx': 0,
        'vocab': 'domain',
        'converter': 'strip_conll12_domain'
      },
      'predicate': {
        'idx': 10,
        'label': True,
        'vocab': 'predicate',
        'converter': 'conll12_binary_predicates'
      },
      'srl': {
        'idx': [14, -1],
        'label': True,
        'vocab': 'srl'
      },
    }

num_epochs = 1
batch_size = 20
is_train = True

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.logging.set_verbosity(tf.logging.INFO)

vocab_names = vocab.make_vocab_files(args.train_file, data_config, args.save_dir)
# vocab_lookup_ops = vocab.create_vocab_lookup_ops(args.train_file, data_config, args)
vocab_lookup_ops = {}


def input_fn():
  with tf.device('/cpu:0'):
    for v in vocab_names:
      num_oov = 1 if data_config[v] else 0
      vocab_lookup_ops[v] = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (args.save_dir, v),
                                                                    num_oov_buckets=num_oov,
                                                                    key_column_index=0)
    if args.word_embedding_file:
      embeddings_name = args.word_embedding_file.split("/")[-1]
      vocab_lookup_ops[embeddings_name] = tf.contrib.lookup.index_table_from_file(args.word_embedding_file,
                                                                                  num_oov_buckets=1,
                                                                                  key_column_index=0,
                                                                                  delimiter=' ')

    # train_data_iterator = dataset.get_data_iterator(args.train_file, data_config, vocab_lookup_ops,
    #                                                 batch_size, num_epochs, is_train)

    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    feature_label_names = [d for d in data_config.keys() if \
                           ('feature' in data_config[d] and data_config[d]['feature']) or \
                           ('label' in data_config[d] and data_config[d]['label'])]


    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: data_generator.conll_data_generator(args.train_file, data_config),
                                             output_shapes=[None, None], output_types=tf.string)

    def map_strings_to_ints(vocab_lookup_ops, data_config, data_names):
      def _mapper(d):
        intmapped = []
        for i, datum_name in enumerate(data_names):
          idx = data_config[datum_name]['idx']
          if isinstance(idx, int):
            if 'vocab' in data_config[datum_name]:
              intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i]), -1))
            else:
              intmapped.append(tf.expand_dims(tf.string_to_number(d[:, i], out_type=tf.int64), -1))
          else:
            last_idx = i + idx[1] if idx[1] > 0 else -1
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:last_idx]))
        return tf.concat(intmapped, axis=-1)

      return _mapper


    def to_input_fn(data_config, data_names):
      def _mapper(d):
        intmapped_feats = []
        intmapped_labels = []
        for i, datum_name in enumerate(data_names):
          if 'feature' in data_config[datum_name] and data_config[datum_name]['feature']:
            intmapped_feats.append(tf.expand_dims(d[:, :, i], -1))
          elif 'label' in data_config[datum_name] and data_config[datum_name]['label']:
            idx = data_config[datum_name]['idx']
            if isinstance(idx, int):
              intmapped_labels.append(tf.expand_dims(d[:, :, i], -1))
            else:
              last_idx = i + idx[1] if idx[1] > 0 else -1
              intmapped_labels.append(d[:, :, i:last_idx])

        labels = tf.concat(intmapped_labels, axis=-1)
        feats = tf.concat(intmapped_feats, axis=-1)
        ret = feats, labels
        return ret

      return _mapper

    # intmap the dataset
    dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names))

    # do batching
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d: tf.shape(d)[0],
                                                                      bucket_boundaries=[20, 30, 50, 80],  # todo: optimal?
                                                                      bucket_batch_sizes=[batch_size] * 5,
                                                                      padded_shapes=dataset.output_shapes))

    dataset = dataset.map(to_input_fn(data_config, feature_label_names))

    # shuffle and expand out epochs if training
    if is_train:
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.shuffle(batch_size * 100)

    # create the iterator
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    a, b = iterator.get_next()
    return a, b


def model_fn(features, labels, mode):

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    hidden_dim = 128
    num_pos_labels = 45
    word_embedding_size = 100
    batch_shape = tf.shape(features)
    batch_size = batch_shape[0]
    batch_seq_len = batch_shape[1]

    tf.logging.log(tf.logging.INFO, "Loading pre-trained word embedding file: %s" % args.word_embedding_file)

    # TODO: np.loadtxt refuses to work for some reason
    # pretrained_embeddings = np.loadtxt(self.args.word_embedding_file, usecols=range(1, word_embedding_size+1))
    pretrained_embeddings = []
    with open(args.word_embedding_file, 'r') as f:
      for line in f:
        split_line = line.split()
        embedding = list(map(float, split_line[1:]))
        pretrained_embeddings.append(embedding)
    pretrained_embeddings = np.array(pretrained_embeddings)
    pretrained_embeddings /= np.std(pretrained_embeddings)
    oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, word_embedding_size], initializer=tf.random_normal_initializer())
    pretrained_embeddings_tensor = tf.get_variable(name="word_embeddings", shape=pretrained_embeddings.shape,
                                            initializer=tf.constant_initializer(pretrained_embeddings))
    word_embeddings_table = tf.concat([pretrained_embeddings_tensor, oov_embedding], axis=0, name="word_embeddings_table")

    words = features[:, :, 0]
    labels = labels[:, :, 0]

    word_embeddings = tf.nn.embedding_lookup(word_embeddings_table, words)

    fwd_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    bwd_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                      inputs=word_embeddings,
                                                      parallel_iterations=50)
    hidden_outputs = tf.concat(axis=2, values=lstm_outputs)
    hidden_outputs_reshape = tf.reshape(hidden_outputs, [-1, 2 * hidden_dim])

    w_o = tf.get_variable(initializer=tf.constant(0.01, shape=[2 * hidden_dim, num_pos_labels]), name="w_o")
    b_o = tf.get_variable(initializer=tf.constant(0.01, shape=[num_pos_labels]), name="b_o")
    scores = tf.nn.xw_plus_b(hidden_outputs_reshape, w_o, b_o, name="scores")
    scores_reshape = tf.reshape(scores, [batch_size, batch_seq_len, num_pos_labels])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores_reshape, labels=labels)

    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    preds = tf.argmax(scores_reshape, -1)
    predictions = {'scores': scores_reshape, 'preds': preds}

    eval_metric_ops = {
      "acc": tf.metrics.accuracy(labels, preds)
    }

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir="model")

estimator.train(input_fn=input_fn, steps=2000)

# np.set_printoptions(threshold=np.inf)
# with tf.Session() as sess:
#   sess.run(tf.tables_initializer())
#   for i in range(3):
#     print(sess.run(input_fn()))

