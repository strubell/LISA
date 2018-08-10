import tensorflow as tf
from tensorflow.estimator import ModeKeys
import nn_utils
import tf_utils


def joint_softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                             transition_params):

  with tf.name_scope('joint_softmax_classifier'):

    # todo pass this as initial proj dim (which is optional)
    projection_dim = model_config['predicate_pred_mlp_size']

    with tf.variable_scope('MLP'):
      mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in joint_softmax_classifier')
      exit(1)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    cross_entropy *= tokens_to_keep
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits
    }

    # now get separate-task predictions for each of the maps we've passed through
    for map_name, label_comp_map in joint_maps.items():
      short_map_name = map_name.split('_to_')[-1]
      label_comp_predictions = tf.nn.embedding_lookup(label_comp_map, predictions)
      output["%s_predictions" % short_map_name] = tf.squeeze(label_comp_predictions, -1)


    return output


def parse_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / num_tokens

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def conditional_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params,
                         dep_rel_mlp, head_rel_mlp, parse_preds_train, parse_preds_eval):

  parse_preds = parse_preds_train if mode == ModeKeys.TRAIN else parse_preds_eval
  with tf.variable_scope('conditional_bilin'):
    logits, _ = nn_utils.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                         parse_preds, hparams.bilinear_dropout)


  print(logits)

  predictions = tf.argmax(logits, -1)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

  n_tokens = tf.reduce_sum(tokens_to_keep)
  loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / n_tokens

  output = {
    'loss': loss,
    'scores': logits,
    'predictions': predictions,
  }

  return output


def srl_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, predicate_preds_train,
                 predicate_preds_eval, predicate_targets, transition_params):
    '''

    :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
    :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
    :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
    :param tokens_to_keep:
    :param predictions:
    :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
    :return:
    '''

    with tf.name_scope('srl_bilinear'):

      input_shape = tf.shape(inputs)
      batch_size = input_shape[0]
      batch_seq_len = input_shape[1]

      predicate_mlp_size = model_config['predicate_mlp_size']
      role_mlp_size = model_config['role_mlp_size']

      predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval

      # (1) project into predicate, role representations
      with tf.variable_scope('MLP'):
        predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
        predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                  predicate_role_mlp[:, :, predicate_mlp_size:]

      # (2) feed through bilinear to obtain scores
      with tf.variable_scope('Bilinear'):
        # gather just the predicates
        # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
        # role mlp: batch x seq_len x role_mlp_size
        # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
        # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
        predicate_gather_indices = tf.where(tf.equal(predicate_preds, 1))
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
        tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                                 [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
        gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)

        # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
        srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                       hparams.bilinear_dropout)
        srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

      # (3) compute loss

      # need to repeat each of these once for each target in the sentence
      mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
      mask = tf.gather_nd(mask_tiled, tf.where(tf.equal(predicate_preds, 1)))

      # now we have k sets of targets for the k frames
      # (p1) f1 f2 f3
      # (p2) f1 f2 f3

      # get all the tags for each token (which is the predicate for a frame), structuring
      # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
      # (p1) f1 f1 f1
      # (p2) f3 f3 f3
      srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

      gold_predicate_counts = tf.reduce_sum(predicate_targets, -1)
      srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

      # num_predicates_in_batch x seq_len
      srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

      predicted_predicate_counts = tf.reduce_sum(predicate_preds, -1)
      srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
      srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

      # num_predicates_in_batch x seq_len
      predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

      seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

      if transition_params is not None and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
        predictions, score = tf.contrib.crf.crf_decode(srl_logits_transposed, transition_params, seq_lens)

      if transition_params is not None and mode == ModeKeys.TRAIN and tf_utils.is_trainable(transition_params):
        # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(srl_logits_transposed,
                                                                              srl_targets_predicted_predicates,
                                                                              seq_lens, transition_params)
        loss = tf.reduce_mean(-log_likelihood)
      else:
        srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
        loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                               onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                               weights=tf.reshape(mask, [-1]),
                                               label_smoothing=hparams.label_smoothing,
                                               reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      output = {
        'loss': loss,
        'predictions': predictions,
        'scores': srl_logits_transposed,
        'targets': srl_targets_gold_predicates,
      }

      return output


dispatcher = {
  'srl_bilinear': srl_bilinear,
  'joint_softmax_classifier': joint_softmax_classifier,
  'parse_bilinear': parse_bilinear,
  'conditional_bilinear': conditional_bilinear,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined output function `%s' % fn_name)
    exit(1)


# need to decide shape/form of train_outputs!
def get_params(mode, model_config, task_map, train_outputs, features, labels, current_outputs, task_labels, num_labels,
               joint_lookup_maps, tokens_to_keep, transition_params, hparams):
  params = {'mode': mode, 'model_config': model_config, 'inputs': current_outputs, 'targets': task_labels,
            'tokens_to_keep': tokens_to_keep, 'num_labels': num_labels, 'transition_params': transition_params,
            'hparams': hparams}
  params_map = task_map['params']
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through
    if 'joint_maps' in param_values:
      params[param_name] = {map_name: joint_lookup_maps[map_name] for map_name in param_values['maps']}
    elif 'label' in param_values:
      params[param_name] = labels[param_values['label']]
    elif 'feature' in param_values:
      params[param_name] = features[param_values['feature']]
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    elif 'layer' in param_values:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
  return params
