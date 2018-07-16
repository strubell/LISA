import tensorflow as tf
import numpy as np
import vocab
import data_generator

class LISAModel:

  def __init__(self, args):
    self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='inputs')
    self.args = args

  def model_fn(self, features, labels, mode):

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      hidden_dim = 128
      num_pos_labels = 45
      word_embedding_size = 100
      batch_shape = tf.shape(features)
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
      pretrained_embeddings /= np.std(pretrained_embeddings)
      oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, word_embedding_size],
                                      initializer=tf.random_normal_initializer())
      pretrained_embeddings_tensor = tf.get_variable(name="word_embeddings", shape=pretrained_embeddings.shape,
                                                     initializer=tf.constant_initializer(pretrained_embeddings))
      word_embeddings_table = tf.concat([pretrained_embeddings_tensor, oov_embedding], axis=0,
                                        name="word_embeddings_table")

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