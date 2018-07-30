import tensorflow as tf
import numpy as np
import constants
import transformer
import nn_utils
import output_fns
import evaluation_fns
from tensorflow.estimator import ModeKeys
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

  def load_transitions(self, transition_statistics, num_classes, vocab_map):
    transition_statistics_np = np.zeros((num_classes, num_classes))
    with open(transition_statistics, 'r') as f:
      for line in f:
        tag1, tag2, prob = line.split("\t")
        transition_statistics_np[vocab_map[tag1], vocab_map[tag2]] = float(prob)
    return transition_statistics_np

  def get_embedding_lookup(self, name, num_embeddings, embedding_size, embedding_values, include_oov,
                           pretrained_fname=None):

    with tf.variable_scope("%s_embeddings" % name):
      initializer = tf.random_normal_initializer()
      if pretrained_fname:
        pretrained_embeddings = self.load_pretrained_embeddings(pretrained_fname)
        initializer = tf.constant_initializer(pretrained_embeddings)

      embedding_table = tf.get_variable(name="embeddings", shape=[num_embeddings, embedding_size],
                                        initializer=initializer)

      if include_oov:
        oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, embedding_size],
                                        initializer=tf.random_normal_initializer())
        embedding_table = tf.concat([embedding_table, oov_embedding], axis=0,
                                    name="embeddings_table")

      return tf.nn.embedding_lookup(embedding_table, embedding_values)

  def load_pretrained_embeddings(self, pretrained_fname):
    tf.logging.log(tf.logging.INFO, "Loading pre-trained embedding file: %s" % pretrained_fname)

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
    # oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, self.model_config['word_embedding_size']],
    #                                 initializer=tf.random_normal_initializer())
    # pretrained_embeddings_tensor = tf.get_variable(name="pretrained_embeddings", shape=pretrained_embeddings.shape,
    #                                                initializer=tf.constant_initializer(pretrained_embeddings))

    # last entry is OOV
    # word_embeddings_table = tf.concat([pretrained_embeddings_tensor, oov_embedding], axis=0,
    #                                   name="word_embeddings_table")
    return pretrained_embeddings

  def model_fn(self, features, mode):

    with tf.variable_scope("LISA", reuse=tf.AUTO_REUSE):

      batch_shape = tf.shape(features)
      batch_size = batch_shape[0]
      batch_seq_len = batch_shape[1]
      layer_config = self.model_config['layers']
      sa_hidden_size = layer_config['head_dim'] * layer_config['num_heads']

      feats = {f: features[:, :, idx] for f, idx in self.feature_idx_map.items()}

      # todo this assumes that word_type is always passed in
      words = feats['word_type']

      # for masking out padding tokens
      tokens_to_keep = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                                tf.ones([batch_size, batch_seq_len]))

      # todo fix masking -- do it in lookup table?
      feats = {f: tf.multiply(tf.cast(tokens_to_keep, tf.int32), v) for f, v in feats.items()}

      labels = {}
      for l, idx in self.label_idx_map.items():
        these_labels = features[:, :, idx[0]:idx[1]] if idx[1] != -1 else features[:, :, idx[0]:]
        these_labels_masked = tf.multiply(these_labels, tf.cast(tf.expand_dims(tokens_to_keep, -1), tf.int32))
        # check if we need to mask another dimension
        if idx[1] == -1:
          last_dim = tf.shape(these_labels)[2]
          this_mask = tf.where(tf.equal(these_labels_masked, constants.PAD_VALUE),
                               tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int32),
                               tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int32))
          these_labels_masked = tf.multiply(these_labels_masked, this_mask)
        else:
          these_labels_masked = tf.squeeze(these_labels_masked, -1)
        labels[l] = these_labels_masked

      # todo concat all the inputs defined in model_config
      inputs_list = []
      for input_name, input_map in self.model_config['inputs'].items():
        # words = feats['word_type']
        input_values = feats[input_name]
        input_embedding_dim = input_map['size']
        input_size = self.vocab.vocab_names_sizes[input_name]
        input_include_oov = self.vocab.oovs[input_name]
        input_embedding_lookup = self.get_embedding_lookup(input_name, input_size, input_embedding_dim,
                                                           input_values, input_include_oov)
        inputs_list.append(input_embedding_lookup)

      current_input = tf.concat(inputs_list, axis=2)

      # words = tf.Print(words, [labels['predicate']], 'predicate labels', summarize=200)


      # todo this is parse specific
      # compute targets adj matrix
      # i1, i2 = tf.meshgrid(tf.range(batch_size), tf.range(batch_seq_len), indexing="ij")
      # idx = tf.stack([i1, i2, labels['parse_head']], axis=-1)
      # adj = tf.scatter_nd(idx, tf.ones([batch_size, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
      # mask2d = tokens_to_keep * tf.transpose(tokens_to_keep)
      # adj = adj * mask2d

      # word_embeddings_table = self.load_pretrained_embeddings(self.args.word_embedding_file)
      # word_embeddings = tf.nn.embedding_lookup(word_embeddings_table, words)
      # current_input = word_embeddings

      # todo will estimators handle dropout for us or do we need to do it on our own?
      input_dropout = self.model_config['input_dropout']
      current_input = tf.nn.dropout(current_input, input_dropout if mode == tf.estimator.ModeKeys.TRAIN else 1.0)

      with tf.variable_scope('project_input'):
        current_input = nn_utils.MLP(current_input, sa_hidden_size, n_splits=1)

      manual_attn = None
      predictions = {}
      eval_metric_ops = {}
      export_outputs = {}
      loss = tf.constant(0.)
      items_to_log = {}

      num_layers = max(self.task_config.keys()) + 1
      tf.logging.log(tf.logging.INFO, "Creating transformer model with %d layers" % num_layers)
      with tf.variable_scope('transformer'):
        for i in range(num_layers):
          with tf.variable_scope('layer%d' % i):
            current_input = transformer.transformer(mode, current_input, tokens_to_keep, layer_config['head_dim'],
                                                    layer_config['num_heads'], layer_config['attn_dropout'],
                                                    layer_config['ff_dropout'], layer_config['prepost_dropout'],
                                                    layer_config['ff_hidden_size'],
                                                    manual_attn)
            if i in self.task_config:
              # todo test a list of tasks for each layer
              for task, task_map in self.task_config[i].items():
                task_labels = labels[task]
                task_vocab_size = self.vocab.vocab_names_sizes[task]

                # Set up CRF / Viterbi transition params if specified
                with tf.variable_scope("crf"):  # to share parameters, change scope here
                  transition_stats_file = task_map['transition_stats'] if 'transition_stats' in task_map else None

                  # todo vocab_lookups not yet initialized -- fix
                  transition_stats = self.load_transitions(transition_stats_file, task_vocab_size,
                                                           self.vocab.vocab_maps[task]) if transition_stats_file else None

                  # create transition parameters if training or decoding with crf/viterbi
                  task_crf = 'crf' in task_map and task_map['crf']
                  task_viterbi_decode = task_crf or 'viterbi' in task_map and task_map['viterbi']
                  transition_params = None
                  if task_viterbi_decode or task_crf:
                    transition_params = tf.get_variable("transitions", [task_vocab_size, task_vocab_size],
                                                        initializer=tf.constant_initializer(transition_stats),
                                                        trainable=task_crf)
                    train_or_decode_str = "training" if task_crf else "decoding"
                    tf.logging.log(tf.logging.INFO, "Created transition params for %s %s" % (train_or_decode_str, task))

                  # if mode == ModeKeys.TRAIN and task_crf:
                  #   transition_params = tf.get_variable("transitions", [task_vocab_size, task_vocab_size],
                  #                                       initializer=tf.constant_initializer(transition_stats))
                  #   tf.logging.log(tf.logging.INFO, "Created transition params for training %s" % task)
                  # elif (mode == ModeKeys.EVAL or mode == ModeKeys.PREDICT) and task_viterbi_decode:
                  #   transition_params = tf.get_variable("transitions", [task_vocab_size, task_vocab_size],
                  #                                       initializer=tf.constant_initializer(transition_stats),
                  #                                       trainable=False)
                  #   tf.logging.log(tf.logging.INFO, "Created transition params for decoding %s" % task)

                output_fn_params = output_fns.get_params(mode, self.model_config, task_map['output_fn'], predictions,
                                                         feats, labels, current_input, task_labels, task_vocab_size,
                                                         self.vocab.joint_label_lookup_maps, tokens_to_keep,
                                                         transition_params)
                task_outputs = output_fns.dispatch(task_map['output_fn']['name'])(**output_fn_params)

                # want task_outputs to have:
                # - predictions
                # - loss
                # - scores
                predictions[task] = task_outputs

                # do the evaluation
                for eval_name, eval_map in task_map['eval_fns'].items():
                  eval_fn_params = evaluation_fns.get_params(task_outputs, eval_map, predictions, feats, labels,
                                                             task_labels, self.vocab.reverse_maps, tokens_to_keep)
                  eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                  eval_metric_ops[eval_name] = eval_result

                # get the individual task loss and apply penalty
                this_task_loss = task_outputs['loss'] * task_map['penalty']

                # this_task_loss = tf.Print(this_task_loss, [this_task_loss], task)

                # log this task's loss
                items_to_log['%s_loss' % task] = this_task_loss

                # add this loss to the overall loss being minimized
                loss += this_task_loss

                # add the predictions to export_outputs
                # todo add un-joint predictions too?
                # predict_output = tf.estimator.export.PredictOutput({'scores': task_outputs['scores'],
                #                                                     'predictions': task_outputs['predictions']})
                # export_outputs['%s_predict' % task] = predict_output

      items_to_log['loss'] = loss

      # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
      # masked_loss = loss * pad_mask
      #
      # loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(pad_mask)

      # todo pass hparams through
      # optimizer = tf.contrib.opt.NadamOptimizer()
      learning_rate = 0.04
      decay_rate = 1.5
      warmup_steps = 8000
      # gradient_clip_norm = 1.0
      gradient_clip_norm = 5.0
      mu = 0.9
      nu = 0.98
      epsilon = 1e-12

      optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=0.0001, beta1=mu, beta2=0.999, epsilon=epsilon)
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
      train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
      # train_op = optimizer.minimize(loss=loss)

      # optimizer = RadamOptimizer(learning_rate=learning_rate, mu=mu, nu=nu, epsilon=epsilon, decay_rate=decay_rate,
      #                            warmup_steps=warmup_steps, gradient_clip_norm=gradient_clip_norm,
      #                            global_step=tf.train.get_global_step())
      # train_op = optimizer.minimize(loss=loss)

      # items_to_log['learning_rate'] = optimizer.learning_rate


      # preds = tf.argmax(scores, -1)
      # predictions = {'scores': scores, 'preds': preds}

      # eval_metric_ops = {
      #   "acc": tf.metrics.accuracy(labels, preds, weights=pad_mask)
      # }

      # export_outputs = {'predict_output': tf.estimator.export.PredictOutput({'scores': scores, 'preds': preds})}

      logging_hook = tf.train.LoggingTensorHook(items_to_log, every_n_iter=20)

      # need to flatten the dict of predictions to make Estimator happy
      flat_predictions = {"%s_%s" % (k1, k2): v2 for k1, v1 in predictions.items() for k2, v2 in v1.items()}

      export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        tf.estimator.export.PredictOutput(flat_predictions)}

      return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, train_op, eval_metric_ops,
                                        training_hooks=[logging_hook], export_outputs=export_outputs)
