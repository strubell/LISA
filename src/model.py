import tensorflow as tf
from tensorflow.estimator import ModeKeys
import constants
import evaluation_fns
import attention_fns
import value_fns
import output_fns
import transformer
import nn_utils
import train_utils
import tf_utils
import util
from lazy_adam_v2 import LazyAdamOptimizer
import bert_util


class LISAModel:

  def __init__(self, hparams, model_config, task_config, attention_config, vocab):
    self.train_hparams = hparams
    self.test_hparams = train_utils.copy_without_dropout(hparams)

    self.model_config = model_config
    self.task_config = task_config
    self.attention_config = attention_config
    # self.feature_idx_map = feature_idx_map
    # self.label_idx_map = label_idx_map
    self.vocab = vocab
    self.frozen_variables = set()

  def hparams(self, mode):
    if mode == ModeKeys.TRAIN:
      return self.train_hparams
    return self.test_hparams

  def get_embedding_table(self, name, embedding_dim, include_oov, pretrained_fname=None, num_embeddings=None):

    with tf.variable_scope("%s_embeddings" % name):
      initializer = tf.random_normal_initializer()
      if pretrained_fname:
        pretrained_embeddings = util.load_pretrained_embeddings(pretrained_fname)
        initializer = tf.constant_initializer(pretrained_embeddings)
        pretrained_num_embeddings, pretrained_embedding_dim = pretrained_embeddings.shape
        if pretrained_embedding_dim != embedding_dim:
          util.fatal_error("Pre-trained %s embedding dim does not match specified dim (%d vs %d)." %
                           (name, pretrained_embedding_dim, embedding_dim))
        if num_embeddings and num_embeddings != pretrained_num_embeddings:
          util.fatal_error("Number of pre-trained %s embeddings does not match specified "
                           "number of embeddings (%d vs %d)." % (name, pretrained_num_embeddings, num_embeddings))
        num_embeddings = pretrained_num_embeddings

      embedding_table = tf.get_variable(name="embeddings", shape=[num_embeddings, embedding_dim],
                                        initializer=initializer)

      if include_oov:
        oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, embedding_dim],
                                        initializer=tf.random_normal_initializer())
        embedding_table = tf.concat([embedding_table, oov_embedding], axis=0,
                                    name="embeddings_table")

      return embedding_table

  def model_fn(self, features, labels, mode):

    # todo can estimators handle dropout for us or do we need to do it on our own?
    hparams = self.hparams(mode)

    print("MODE(%s) features" % str(mode), features)

    with tf.variable_scope("LISA"):#, reuse=tf.AUTO_REUSE):

      predictions = {}
      eval_metric_ops = {}
      export_outputs = {}
      loss = tf.constant(0.)
      items_to_log = {}

      # todo this assumes that word is always passed in, and that it has the same shape as all the other stuff
      words = features['word']

      # batch_shape = tf.shape(words)
      # batch_size = batch_shape[0]
      # batch_seq_len = batch_shape[1]
      layer_config = self.model_config['layers']
      sa_hidden_size = layer_config['head_dim'] * layer_config['num_heads']

      # for masking out padding tokens
      tokens_to_keep = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros_like(words, dtype=tf.float32),
                                tf.ones_like(words, dtype=tf.float32))

      def get_mask(t):
        return tf.where(tf.equal(t, constants.PAD_VALUE), tf.zeros_like(t), tf.ones_like(t))

      # Zero-mask features
      named_features = {f: tf.multiply(get_mask(v), v) for f, v in features.items()}

      # Extract named labels from monolithic "labels" input, and mask them
      # todo: why is this even necessary?
      named_labels = None
      if mode != ModeKeys.PREDICT:
        named_labels = {f: tf.multiply(get_mask(v), v) for f, v in labels.items()}

        # for l, these_labels in named_labels.items():
        #   this_shape = these_labels.get_shape().as_list()
        #   if len(this_shape) == 2:
        #     these_labels_masked = tf.multiply(these_labels, tf.cast(tokens_to_keep, tf.int64))
        #   # check if we need to mask another dimension
        #   elif len(this_shape) == 3:
        #     last_dim = tf.shape(these_labels)[2]
        #     this_mask = tf.where(tf.equal(these_labels, constants.PAD_VALUE),
        #                          tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int64),
        #                          tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int64))
        #     these_labels_masked = tf.multiply(these_labels, this_mask)
        #   named_labels[l] = these_labels_masked

      # load transition parameters
      transition_stats = util.load_transition_params(self.task_config, self.vocab)

      # Create embeddings tables, loading pre-trained or BERT if specified
      embeddings = {}
      for embedding_name, embedding_map in self.model_config['embeddings'].items():
        embedding_dim = embedding_map['embedding_dim']
        if 'pretrained_embeddings' in embedding_map:
          input_pretrained_embeddings = embedding_map['pretrained_embeddings']
          include_oov = True
          embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                     pretrained_fname=input_pretrained_embeddings)
          embeddings[embedding_name] = embedding_table
        elif 'bert_embeddings' in embedding_map:

          # todo don't hardcode this lookup, specify it in the config
          bpe_sentences = named_features['word_bpe']
          bert_dir = embedding_map['bert_embeddings']
          bpe_lens = named_features['word_bpe_lens']

          bert_embedded_tokens, bert_vars, bert_weights_l2 = bert_util.get_bert_embeddings(bert_dir, bpe_sentences,
                                                                                           bpe_lens, self.vocab.vocab_maps)

          loss += bert_weights_l2

          # don't update bert parameters
          # todo don't hardcode to not update bert
          self.frozen_variables.update(bert_vars)

          # add 'bert_words' to named_features
          named_features['bert_words'] = bert_embedded_tokens

        else:
          num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
          include_oov = self.vocab.oovs[embedding_name]
          embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                     num_embeddings=num_embeddings)
          embeddings[embedding_name] = embedding_table
        tf.logging.info("Created embeddings for '%s'." % embedding_name)

      # Set up model inputs
      inputs_list = []
      for input_name in self.model_config['inputs']:
        input_values = named_features[input_name]
        if 'bert' in input_name:
          inputs_list.append(input_values)
        else:
          input_embedding_lookup = tf.nn.embedding_lookup(embeddings[input_name], input_values)
          inputs_list.append(input_embedding_lookup)
        tf.logging.info("Added %s to inputs list." % input_name)
      current_input = tf.concat(inputs_list, axis=2)
      current_input = tf.nn.dropout(current_input, hparams.input_dropout)

      with tf.variable_scope('project_input'):
        current_input = nn_utils.MLP(current_input, sa_hidden_size, n_splits=1)

      num_layers = max(self.task_config.keys()) + 1
      tf.logging.info("Creating transformer model with %d layers" % num_layers)
      with tf.variable_scope('transformer'):
        current_input = transformer.add_timing_signal_1d(current_input)
        for i in range(num_layers):
          with tf.variable_scope('layer%d' % i):

            special_attn = []
            special_values = []
            if i in self.attention_config:

              this_layer_attn_config = self.attention_config[i]

              if 'attention_fns' in this_layer_attn_config:
                for attn_fn, attn_fn_map in this_layer_attn_config['attention_fns'].items():
                  attention_fn_params = attention_fns.get_params(mode, attn_fn_map, predictions, named_features, named_labels)
                  this_special_attn = attention_fns.dispatch(attn_fn_map['name'])(**attention_fn_params)
                  special_attn.append(this_special_attn)

              if 'value_fns' in this_layer_attn_config:
                for value_fn, value_fn_map in this_layer_attn_config['value_fns'].items():
                  value_fn_params = value_fns.get_params(mode, value_fn_map, predictions, named_features, named_labels, embeddings)
                  this_special_values = value_fns.dispatch(value_fn_map['name'])(**value_fn_params)
                  special_values.append(this_special_values)

            current_input = transformer.transformer(current_input, tokens_to_keep, layer_config['head_dim'],
                                                    layer_config['num_heads'], hparams.attn_dropout,
                                                    hparams.ff_dropout, hparams.prepost_dropout,
                                                    layer_config['ff_hidden_size'], special_attn, special_values)
            if i in self.task_config:

              # if normalization is done in layer_preprocess, then it should also be done
              # on the output, since the output can grow very large, being the sum of
              # a whole stack of unnormalized layer outputs.
              current_input = nn_utils.layer_norm(current_input)

              # todo test a list of tasks for each layer
              for task, task_map in self.task_config[i].items():
                # task_labels = named_labels[task]
                task_vocab_size = self.vocab.vocab_names_sizes[task] if task in self.vocab.vocab_names_sizes else -1

                # Set up CRF / Viterbi transition params if specified
                with tf.variable_scope("crf"):  # to share parameters, change scope here
                  # transition_stats_file = task_map['transition_stats'] if 'transition_stats' in task_map else None
                  task_transition_stats = transition_stats[task] if task in transition_stats else None

                  # create transition parameters if training or decoding with crf/viterbi
                  task_crf = 'crf' in task_map and task_map['crf']
                  task_viterbi_decode = task_crf or 'viterbi' in task_map and task_map['viterbi']
                  transition_params = None
                  if task_viterbi_decode or task_crf:
                    transition_params = tf.get_variable("transitions", [task_vocab_size, task_vocab_size],
                                                        initializer=tf.constant_initializer(task_transition_stats),
                                                        trainable=task_crf)
                    train_or_decode_str = "training" if task_crf else "decoding"
                    tf.logging.info("Created transition params for %s %s" % (train_or_decode_str, task))

                task_labels = named_labels[task] if mode != ModeKeys.PREDICT else None
                output_fn_params = output_fns.get_params(mode, self.model_config, task_map['output_fn'], predictions,
                                                         named_features, current_input, task_vocab_size,
                                                         self.vocab.joint_label_lookup_maps, tokens_to_keep,
                                                         transition_params, hparams, named_labels, task_labels)
                task_outputs = output_fns.dispatch(task_map['output_fn']['name'])(**output_fn_params)

                # want task_outputs to have:
                # - predictions
                # - scores
                # - probabilities
                # - loss (if training)
                predictions[task] = task_outputs

                if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:

                  # get the individual task loss and apply penalty
                  this_task_loss = task_outputs['loss'] * task_map['penalty']

                  # log this task's loss
                  items_to_log['%s_loss' % task] = this_task_loss

                  # add this loss to the overall loss being minimized
                  loss += this_task_loss

                  if mode == ModeKeys.EVAL:
                    # do the evaluation
                    for eval_name, eval_map in task_map['eval_fns'].items():
                      eval_fn_params = evaluation_fns.get_params(mode, task_outputs, eval_map, predictions,
                                                                 named_features,
                                                                 named_labels, task_labels, self.vocab.reverse_maps,
                                                                 tokens_to_keep)
                      eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                      eval_metric_ops[eval_name] = eval_result

      if mode == ModeKeys.PREDICT:
        # need to flatten the dict of predictions to make Estimator happy
        flat_predictions = {"%s_%s" % (k1, k2): v2 for k1, v1 in predictions.items() for k2, v2 in v1.items()}
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            tf.estimator.export.PredictOutput(flat_predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=flat_predictions, export_outputs=export_outputs)

      # set up moving average variables
      assign_moving_averages_dep = tf.no_op()
      if hparams.moving_average_decay > 0.:
        moving_averager = tf.train.ExponentialMovingAverage(hparams.moving_average_decay, zero_debias=True,
                                                            num_updates=tf.train.get_global_step())
        moving_average_op = moving_averager.apply(train_utils.get_vars_for_moving_average(hparams.average_norms))
        # tf.logging.info("Using moving average for variables: %s" % str([v.name for v in tf.trainable_variables()])

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, moving_average_op)

        # use moving averages of variables if evaluating
        assign_moving_averages_dep = tf.cond(tf.equal(mode, ModeKeys.TRAIN),
                                             lambda: tf.no_op(),
                                             lambda: nn_utils.set_vars_to_moving_average(moving_averager))

      with tf.control_dependencies([assign_moving_averages_dep]):

        items_to_log['loss'] = loss

        # get learning rate w/ decay
        this_step_lr = train_utils.learning_rate(hparams, tf.train.get_global_step())
        items_to_log['lr'] = this_step_lr

        # optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=this_step_lr, beta1=hparams.beta1,
        #                                              beta2=hparams.beta2, epsilon=hparams.epsilon)
        optimizer = LazyAdamOptimizer(learning_rate=this_step_lr, beta1=hparams.beta1,
                                      beta2=hparams.beta2, epsilon=hparams.epsilon,
                                      use_nesterov=hparams.use_nesterov)
        vars_to_train = [v for v in tf.trainable_variables() if v not in self.frozen_variables]
        gradients, variables = zip(*optimizer.compute_gradients(loss, var_list=vars_to_train))
        gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

        # export_outputs = {'predict_output': tf.estimator.export.PredictOutput({'scores': scores, 'preds': preds})}

        logging_hook = tf.train.LoggingTensorHook(items_to_log, every_n_iter=20)

        tf.logging.info("Created model with %d trainable parameters" % tf_utils.get_num_parameters(vars_to_train))

        if mode == ModeKeys.EVAL:
          return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops, train_op=train_op,
                                          training_hooks=[logging_hook])
