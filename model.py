import tensorflow as tf
import numpy as np
import constants
import transformer
import nn_utils
import output_fns
import evaluation_fns
from radam_optimizer import RadamOptimizer


class LISAModel:

  def __init__(self, args, model_config, task_config, feature_idx_map, label_idx_map, vocab):
    self.args = args
    self.model_config = model_config
    self.task_config = task_config
    self.feature_idx_map = feature_idx_map
    self.label_idx_map = label_idx_map
    # self.joint_label_lookup_maps = joint_label_lookup_maps
    # self.label_vocab_sizes = label_vocab_sizes
    self.vocab = vocab


  def load_pretrained_embeddings(self):
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
    oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, self.model_config['word_embedding_size']],
                                    initializer=tf.random_normal_initializer())
    pretrained_embeddings_tensor = tf.get_variable(name="word_embeddings", shape=pretrained_embeddings.shape,
                                                   initializer=tf.constant_initializer(pretrained_embeddings))

    # last entry is OOV
    word_embeddings_table = tf.concat([pretrained_embeddings_tensor, oov_embedding], axis=0,
                                      name="word_embeddings_table")
    return word_embeddings_table

  def model_fn(self, features, mode):

    with tf.variable_scope("LISA", reuse=tf.AUTO_REUSE):

      batch_shape = tf.shape(features)
      batch_size = batch_shape[0]
      batch_seq_len = batch_shape[1]
      layer_config = self.model_config['layers']
      sa_hidden_size = layer_config['head_dim'] * layer_config['num_heads']

      feats = {f: features[:, :, idx] for f, idx in self.feature_idx_map.items()}
      labels = {l: features[:, :, idx] for l, idx in self.label_idx_map.items()}

      words = feats['word']

      # for masking out padding tokens
      tokens_to_keep = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                                tf.ones([batch_size, batch_seq_len]))

      words *= tf.cast(tokens_to_keep, tf.int32)


      # todo this is parse specific
      # compute targets adj matrix
      # i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(batch_seq_len), indexing="ij")
      # idx = tf.stack([i1, i2, labels['parse_head']], axis=-1)
      # adj = tf.scatter_nd(idx, tf.ones([batch_size, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
      # mask2d = tokens_to_keep * tf.transpose(tokens_to_keep)
      # adj = adj * mask2d

      word_embeddings_table = self.load_pretrained_embeddings()
      word_embeddings = tf.nn.embedding_lookup(word_embeddings_table, words)

      current_input = word_embeddings
      current_input = tf.nn.dropout(current_input, self.model_config['input_dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0)

      with tf.variable_scope('project_input'):
        current_input = nn_utils.MLP(current_input, sa_hidden_size, n_splits=1)

      manual_attn = None
      predictions = {}
      eval_metric_ops = {}
      export_outputs = {}
      loss = tf.constant(0.)
      items_to_log = {'loss': loss}
      with tf.variable_scope('transformer'):

        for i in range(self.model_config['num_layers']):
          with tf.variable_scope('layer%d' % i):
            current_input = transformer.transformer(mode, current_input, tokens_to_keep, layer_config['head_dim'],
                                                    layer_config['num_heads'], layer_config['attn_dropout'],
                                                    layer_config['ff_dropout'], layer_config['prepost_dropout'],
                                                    layer_config['ff_hidden_size'],
                                                    manual_attn)
            if i in self.task_config:
              for task, task_map in self.task_config[i].items():
                task_labels = labels[task] * tf.cast(tokens_to_keep, tf.int32)
                output_fn_params = output_fns.get_params(self.model_config, task_map['output_fn'], predictions,
                                                         current_input, task_labels, self.vocab.vocab_names_sizes[task],
                                                         self.vocab.joint_label_lookup_maps, tokens_to_keep)
                task_outputs = output_fns.dispatch(task_map['output_fn']['name'])(**output_fn_params)

                # want task_outputs to have:
                # - predictions
                # - loss
                # - scores
                predictions[task] = task_outputs

                # do the evaluation
                eval_fn_params = evaluation_fns.get_params(task_outputs['predictions'], task_map, predictions,
                                                           task_labels, tokens_to_keep)
                eval_result = evaluation_fns.dispatch(task_map['eval_fn']['name'])(**eval_fn_params)
                eval_metric_ops['%s_%s' % (task, task_map['eval_fn']['name'])] = eval_result

                # get the individual task loss and apply penalty
                this_task_loss = task_outputs['loss'] * task_map['penalty']

                this_task_loss = tf.Print(this_task_loss, [this_task_loss], task)

                # log this task's loss
                items_to_log['%s_loss' % task] = this_task_loss

                # add this loss to the overall loss being minimized
                loss += this_task_loss

                # add the predictions to export_outputs
                # todo add un-joint predictions too?
                predict_output = tf.estimator.export.PredictOutput({'scores': task_outputs['scores'],
                                                                    'predictions': task_outputs['predictions']})
                export_outputs['%s_predict' % task] = predict_output



      # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
      # masked_loss = loss * pad_mask
      #
      # loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(pad_mask)

      # todo
      # optimizer = tf.contrib.opt.NadamOptimizer()
      learning_rate = 0.04
      decay_rate = 1.5
      warmup_steps = 8000
      gradient_clip_norm = 1.0
      mu = 0.9
      nu = 0.98
      epsilon = 1e-12
      # optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate, beta1=mu, beta2=nu, epsilon=epsilon)
      optimizer = RadamOptimizer(learning_rate=learning_rate, mu=mu, nu=nu, epsilon=epsilon, decay_rate=decay_rate,
                                 warmup_steps=warmup_steps, gradient_clip_norm=gradient_clip_norm,
                                 global_step=tf.train.get_global_step())
      train_op = optimizer.minimize(loss=loss)
      # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())


      # preds = tf.argmax(scores, -1)
      # predictions = {'scores': scores, 'preds': preds}

      # eval_metric_ops = {
      #   "acc": tf.metrics.accuracy(labels, preds, weights=pad_mask)
      # }

      # export_outputs = {'predict_output': tf.estimator.export.PredictOutput({'scores': scores, 'preds': preds})}

      logging_hook = tf.train.LoggingTensorHook(items_to_log, every_n_iter=10)

      flat_predictions = {"%s_%s" % (k1, k2): v2 for k1, v1 in predictions.items() for k2, v2 in v1.items()}

      return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, train_op, eval_metric_ops,
                                        training_hooks=[logging_hook], export_outputs=export_outputs)
