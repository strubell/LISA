import tensorflow as tf
import nn_utils


def joint_softmax_classifier(model_config, input, targets, num_labels, tokens_to_keep, joint_maps):

  predicate_pred_mlp_size = model_config['predicate_pred_mlp_size']

  with tf.variable_scope('MLP'):
    mlp = nn_utils.MLP(input, predicate_pred_mlp_size, n_splits=1)
  with tf.variable_scope('Classifier'):
    logits = nn_utils.MLP(mlp, num_labels, n_splits=1)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

  # squeezed_mask = tf.squeeze(tokens_to_keep, -1)
  # int_mask = tf.cast(squeezed_mask, tf.int32)

  cross_entropy *= tf.squeeze(tokens_to_keep, -1)
  loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)

  predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
  # predictions *= int_mask

  output = {
    'loss': loss,
    'predictions': predictions,
    'scores': logits
  }

  # now get separate-task predictions for each of the maps we've passed through
  for map_name, label_comp_map in joint_maps.items():
    label_comp_predictions = tf.nn.embedding_lookup(label_comp_map, predictions)
    output["%s_predictions" % map_name] = tf.squeeze(label_comp_predictions, -1)

  return output


def srl_bilinear(model_config, input, targets, num_labels, tokens_to_keep, predicate_preds, transition_params=None):
    '''

    :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
    :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
    :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len]
    :param tokens_to_keep:
    :param predictions:
    :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
    :return:
    '''

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    predicate_mlp_size = model_config['predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size']
    label_smoothing = model_config['label_smoothing']
    # todo pass this in
    bilin_keep_prob = 1.0

    # (1) project into predicate, role representations
    with tf.variable_scope('MLP'):
      predicate_role_mlp = nn_utils.MLP(input, predicate_mlp_size + role_mlp_size, n_splits=1)
      predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                predicate_role_mlp[:, :, predicate_mlp_size:]

    # (2) feed through bilinear to obtain scores
    with tf.variable_scope('Bilinear'):
      # gather just the predicates
      # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
      # role mlp: batch x seq_len x role_mlp_size
      # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
      # i.e. a (num_predicates_in_batch x bucket_size x role_mlp_size) tensor
      predicate_gather_indices = tf.where(tf.equal(predicate_preds, 1))
      gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
      tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                               [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
      gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)

      # now multiply them together to get (num_predicates_in_batch x bucket_size x num_srl_classes) tensor of scores
      srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels, bilin_keep_prob)
      srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

    # (3) compute loss

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tf.squeeze(tokens_to_keep, -1), [1, batch_seq_len]),
                            [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, tf.where(tf.equal(predicate_preds, 1)))
    count = tf.cast(tf.count_nonzero(mask), tf.float32)

    # now we have k sets of targets for the k frames
    # (t1) f1 f2 f3
    # (t2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming t1 and t2 are predicates for f1 and f3, respectively):
    # (t1) f1 f1 f1
    # (t2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    # num_triggers_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    predicate_counts = tf.reduce_sum(predicate_preds, -1)

    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(predicate_counts, [-1])))

    srl_targets = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    if transition_params is not None:
      seq_lens = tf.reduce_sum(mask, 1)
      # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
      log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(srl_logits_transposed, srl_targets,
                                                                            seq_lens,
                                                                            transition_params=transition_params)
      loss = tf.reduce_mean(-log_likelihood)
    else:
      if label_smoothing > 0:
        srl_targets_onehot = tf.one_hot(indices=srl_targets, depth=num_labels, axis=-1)
        loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                               onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                               weights=tf.reshape(mask, [-1]),
                                               label_smoothing=label_smoothing,
                                               reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=srl_logits_transposed, labels=srl_targets)
        cross_entropy *= mask
        loss = tf.cond(tf.equal(count, 0.), lambda: tf.constant(0.), lambda: tf.reduce_sum(cross_entropy) / count)

    # correct = tf.reduce_sum(tf.cast(tf.equal(predictions, srl_targets), tf.float32))
    # probabilities = tf.nn.softmax(srl_logits_transposed)

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': srl_logits_transposed
    }

    return output


dispatcher = {
  'srl_bilinear': srl_bilinear,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined output function `%s' % fn_name)
    exit(1)


# need to decide shape/form of train_outputs!
def get_params(model_config, task_map, train_outputs, current_outputs, task_labels, num_labels, joint_lookup_maps,
               tokens_to_keep):
  params = {'model_config': model_config, 'inputs': current_outputs, 'targets': task_labels,
            'tokens_to_keep': tokens_to_keep, 'num_labels': num_labels}
  params_map = task_map['params']
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through
    if 'maps' in param_values:
      print(param_values['maps'])
      print(joint_lookup_maps['joint_pos_predicate_to_predicate'])
      params[param_name] = {joint_lookup_maps[map_name] for map_name in param_values['maps']}
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    else:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]

  return params
