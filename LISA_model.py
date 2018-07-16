import tensorflow as tf
import numpy as np
import vocab
import data_generator

class LISAModel:

  def __init__(self, args):
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='inputs')
    self.args = args

    self.data_config = {
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


  def input_fn(self):

    num_epochs = 1
    batch_size = 20
    is_train = True

    # vocab_lookup_ops = vocab.create_vocab_lookup_ops(args.train_file, data_config, args)
    vocab_names = vocab.make_vocab_files(self.args.train_file, self.data_config, self.args.save_dir)
    vocab_lookup_ops = {}

    with tf.device('/cpu:0'):
      for v in vocab_names:
        num_oov = 1 if self.data_config[v] else 0
        vocab_lookup_ops[v] = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (self.args.save_dir, v),
                                                                      num_oov_buckets=num_oov,
                                                                      key_column_index=0)
      if self.args.word_embedding_file:
        embeddings_name = self.args.word_embedding_file.split("/")[-1]
        vocab_lookup_ops[embeddings_name] = tf.contrib.lookup.index_table_from_file(self.args.word_embedding_file,
                                                                                    num_oov_buckets=1,
                                                                                    key_column_index=0,
                                                                                    delimiter=' ')

    # train_data_iterator = dataset.get_data_iterator(args.train_file, data_config, vocab_lookup_ops,
    #                                                 batch_size, num_epochs, is_train)
    with tf.device('/cpu:0'):

      # get the names of data fields in data_config that correspond to features or labels,
      # and thus that we want to load into batches
      feature_label_names = [d for d in self.data_config.keys() if \
                             ('feature' in self.data_config[d] and self.data_config[d]['feature']) or \
                             ('label' in self.data_config[d] and self.data_config[d]['label'])]

      # get the dataset
      dataset = tf.data.Dataset.from_generator(lambda: data_generator.conll_data_generator(self.args.train_file, self.data_config),
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

      # intmap the dataset
      dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, self.data_config, feature_label_names))

      # do batching
      dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d: tf.shape(d)[0],
                                                                        bucket_boundaries=[20, 30, 50, 80],
                                                                        # todo: optimal?
                                                                        bucket_batch_sizes=[batch_size] * 5,
                                                                        padded_shapes=dataset.output_shapes))

      # shuffle and expand out epochs if training
      if is_train:
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.shuffle(batch_size * 100)

      # create the iterator
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return iterator.get_next

  def model_fn(self, features, labels, mode):

    hidden_dim = 128
    num_pos_labels = 45
    word_embedding_size = 100
    batch_shape = tf.shape(self.inputs)
    batch_size = batch_shape[0]
    batch_seq_len = batch_shape[1]

    tf.logging.log(tf.logging.INFO, "Loading pre-trained word embedding file: %s" % self.args.word_embedding_file)

    # TODO: np.loadtxt refuses to work for some reason
    # pretrained_embeddings = np.loadtxt(self.args.word_embedding_file, usecols=range(1, word_embedding_size+1))
    pretrained_embeddings = []
    with open(self.args.word_embedding_file, 'r') as f:
      for line in f:
        split_line = line.split()
        embedding = list(map(float, split_line[1:]))
        pretrained_embeddings.append(embedding)
    pretrained_embeddings = np.array(pretrained_embeddings)
    word_embeddings_table = tf.get_variable(name="word_embeddings", shape=pretrained_embeddings.shape,
                                            initializer=tf.constant_initializer(pretrained_embeddings))

    words = self.inputs[:, :, 1]
    labels = self.inputs[:, :, 2]

    word_embeddings = tf.nn.embedding_lookup(word_embeddings_table, words)

    fwd_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    bwd_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwd_cell, cell_bw=bwd_cell, dtype=tf.float32,
                                                      inputs=word_embeddings,
                                                      parallel_iterations=50)
    hidden_outputs = tf.concat(axis=2, values=lstm_outputs)
    hidden_outputs_reshape = tf.reshape(hidden_outputs, [-1, 2 * hidden_dim])

    w_o = tf.get_variable(initializer=tf.constant(0.01, shape=[2 * hidden_dim, num_pos_labels]), name="b_o")
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