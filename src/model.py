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
import bert.modeling


class LISAModel:

  def __init__(self, hparams, model_config, task_config, attention_config, feature_idx_map, label_idx_map,
               vocab):
    self.train_hparams = hparams
    self.test_hparams = train_utils.copy_without_dropout(hparams)

    self.model_config = model_config
    self.task_config = task_config
    self.attention_config = attention_config
    self.feature_idx_map = feature_idx_map
    self.label_idx_map = label_idx_map
    self.vocab = vocab

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

    intmapped_feats, sentences = features

    with tf.variable_scope("LISA", reuse=tf.AUTO_REUSE):

      batch_shape = tf.shape(intmapped_feats)
      batch_size = batch_shape[0]
      batch_seq_len = batch_shape[1]
      layer_config = self.model_config['layers']
      sa_hidden_size = layer_config['head_dim'] * layer_config['num_heads']

      print("feature idx map", self.feature_idx_map)
      print("label idx map", self.label_idx_map)
      print("intmapped_feats", intmapped_feats)
      print("sentences", sentences)
      print("labels", labels)

      named_features = {f: tf.squeeze(intmapped_feats[:, :, idx[0]:idx[1]], -1) if idx[1] != -1 else intmapped_feats[:, :, idx[0]:]
               for f, idx in self.feature_idx_map.items()}

      # todo this assumes that word_type is always passed in
      words = named_features['word_type']

      # for masking out padding tokens
      tokens_to_keep = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                                tf.ones([batch_size, batch_seq_len]))

      # Extract named features from monolithic "features" input
      named_features = {f: tf.multiply(tf.cast(tokens_to_keep, tf.int32), v) for f, v in named_features.items()}

      # Extract named labels from monolithic "features" input, and mask them
      # todo fix masking -- is it even necessary?
      named_labels = {}
      for l, idx in self.label_idx_map.items():
        these_labels = labels[:, :, idx[0]:idx[1]] if idx[1] != -1 else labels[:, :, idx[0]:]
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
        named_labels[l] = these_labels_masked

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
          # tokenize into word pieces
          bert_dir = embedding_map['bert_embeddings']
          embedding_vocab_name = bert_dir + "/vocab.txt"
          bert_config = bert_dir + "/bert_config.json"
          # bert_checkpoint = bert_dir + ""
          # bert_vocab = bert_dir + "/vocab.txt"
          # bert_cased = 'cased' in bert_dir
          bpe_words = sentences

          # d = tf.data.Dataset.from_tensor_slices({
          #   "unique_ids":
          #     tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
          #   "input_ids":
          #     tf.constant(
          #       all_input_ids, shape=[num_examples, seq_length],
          #       dtype=tf.int32),
          #   "input_mask":
          #     tf.constant(
          #       all_input_mask,
          #       shape=[num_examples, seq_length],
          #       dtype=tf.int32),
          #   "input_type_ids":
          #     tf.constant(
          #       all_input_type_ids,
          #       shape=[num_examples, seq_length],
          #       dtype=tf.int32),
          # })
          bert_model = bert.modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=bpe_words,
            input_mask=tf.where(tf.not_equal(bpe_words, 0)),
            token_type_ids=tf.zeros_like(bpe_words),
            use_one_hot_embeddings=False)

          tvars = tf.trainable_variables()
          assignment_map, initialized_variable_names = bert.modeling.get_assignment_map_from_checkpoint(tvars, bert_dir)

          tf.train.init_from_checkpoint(bert_dir, assignment_map)

          tf.logging.info("**** BERT Trainable Variables ****")
          for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
              init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

          # list of [batch_size, seq_length, hidden_size]
          bert_embeddings = bert_model.all_encoder_layers()
          for bert_layer_idx, bert_layer_output in enumerate(bert_embeddings):
            # todo this is wrong, pretty sure these should sum to 1
            layer_weight = tf.get_variable("bert_layer_weight_%d" % bert_layer_idx)
            bert_embeddings[bert_layer_idx] = bert_layer_output * layer_weight
          bert_embeddings_concat = tf.concat(bert_embeddings, axis=-1)
          bert_embeddings_avg = tf.reduce_sum(bert_embeddings_concat, -1)

          # use lens to combine bpe reps back into token reps
          bpe_lens = named_features['word_bpe_lens']
          max_bpe_len = tf.reduce_max(bpe_lens)
          scatter_indices = tf.where(tf.sequence_mask(tf.reshape(bpe_lens, [-1])))
          bert_reps_scatter = tf.scatter_nd(scatter_indices, bert_embeddings_avg, [batch_size*batch_seq_len, max_bpe_len])

          bert_tokens = tf.reshape(tf.reduce_mean(bert_reps_scatter, axis=-1), [batch_size, batch_seq_len, -1])

          # add 'bert_words' to named_features
          named_features['bert_words'] = bert_tokens

          # dummy thing
          # num_embeddings = self.vocab.vocab_names_sizes[embedding_vocab_name]
          # print(num_embeddings, embedding_dim)
          # include_oov = False
          # embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
          #                                            num_embeddings=num_embeddings)

        else:
          num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
          include_oov = self.vocab.oovs[embedding_name]
          embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                     num_embeddings=num_embeddings)
          embeddings[embedding_name] = embedding_table
        tf.logging.log(tf.logging.INFO, "Created embeddings for '%s'." % embedding_name)

      # Set up model inputs
      inputs_list = []
      for input_name in self.model_config['inputs']:
        input_values = named_features[input_name]
        if 'bert' in input_name:
          inputs_list.append(input_values)
        else:  
          input_embedding_lookup = tf.nn.embedding_lookup(embeddings[input_name], input_values)
          inputs_list.append(input_embedding_lookup)
        tf.logging.log(tf.logging.INFO, "Added %s to inputs list." % input_name)
      current_input = tf.concat(inputs_list, axis=2)
      current_input = tf.nn.dropout(current_input, hparams.input_dropout)

      with tf.variable_scope('project_input'):
        current_input = nn_utils.MLP(current_input, sa_hidden_size, n_splits=1)

      predictions = {}
      eval_metric_ops = {}
      export_outputs = {}
      loss = tf.constant(0.)
      items_to_log = {}

      num_layers = max(self.task_config.keys()) + 1
      tf.logging.log(tf.logging.INFO, "Creating transformer model with %d layers" % num_layers)
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
                task_labels = labels[task]
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
                    tf.logging.log(tf.logging.INFO, "Created transition params for %s %s" % (train_or_decode_str, task))

                output_fn_params = output_fns.get_params(mode, self.model_config, task_map['output_fn'], predictions,
                                                         feats, labels, current_input, task_labels, task_vocab_size,
                                                         self.vocab.joint_label_lookup_maps, tokens_to_keep,
                                                         transition_params, hparams)
                task_outputs = output_fns.dispatch(task_map['output_fn']['name'])(**output_fn_params)

                # want task_outputs to have:
                # - predictions
                # - loss
                # - scores
                # - probabilities
                predictions[task] = task_outputs

                # do the evaluation
                for eval_name, eval_map in task_map['eval_fns'].items():
                  eval_fn_params = evaluation_fns.get_params(task_outputs, eval_map, predictions, named_features, named_labels,
                                                             task_labels, self.vocab.reverse_maps, tokens_to_keep)
                  eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                  eval_metric_ops[eval_name] = eval_result

                # get the individual task loss and apply penalty
                this_task_loss = task_outputs['loss'] * task_map['penalty']

                # log this task's loss
                items_to_log['%s_loss' % task] = this_task_loss

                # add this loss to the overall loss being minimized
                loss += this_task_loss

      # set up moving average variables
      assign_moving_averages_dep = tf.no_op()
      if hparams.moving_average_decay > 0.:
        moving_averager = tf.train.ExponentialMovingAverage(hparams.moving_average_decay, zero_debias=True,
                                                            num_updates=tf.train.get_global_step())
        moving_average_op = moving_averager.apply(train_utils.get_vars_for_moving_average(hparams.average_norms))
        # tf.logging.log(tf.logging.INFO,
        #                "Using moving average for variables: %s" % str([v.name for v in tf.trainable_variables()])

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
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())

        # export_outputs = {'predict_output': tf.estimator.export.PredictOutput({'scores': scores, 'preds': preds})}

        logging_hook = tf.train.LoggingTensorHook(items_to_log, every_n_iter=20)

        # need to flatten the dict of predictions to make Estimator happy
        flat_predictions = {"%s_%s" % (k1, k2): v2 for k1, v1 in predictions.items() for k2, v2 in v1.items()}

        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                          tf.estimator.export.PredictOutput(flat_predictions)}

        tf.logging.log(tf.logging.INFO,
                       "Created model with %d trainable parameters" % tf_utils.get_num_trainable_parameters())

        return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, train_op, eval_metric_ops,
                                          training_hooks=[logging_hook], export_outputs=export_outputs)
