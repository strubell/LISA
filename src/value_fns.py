import tensorflow as tf


def label_attention(mode, train_label_scores, eval_label_scores, label_embeddings):
  with tf.variable_scope('label_attention_value_fn'):

    embeddings_shape = label_embeddings.get_shape()
    vocab_size = embeddings_shape[0]
    label_embedding_dim = embeddings_shape[1]
    input_shape = tf.shape(train_label_scores)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    label_scores = train_label_scores if mode == tf.estimator.ModeKeys.TRAIN else eval_label_scores

    # check whether this thing is actually scores or if it's predictions, and needs
    # to be expanded out to one-hot scores. If it's actually scores, dims should be
    # batch x batch_seq_len x num_classes, and thus rank should be 3
    if len(label_scores.get_shape()) < 3:
      label_scores = tf.one_hot(label_scores, vocab_size)

    softmax_label_scores = tf.nn.softmax(label_scores)
    softmax_label_scores = tf.reshape(softmax_label_scores, [-1, vocab_size])
    label_embeddings = tf.reshape(label_embeddings, [vocab_size, label_embedding_dim])
    averaged = tf.matmul(softmax_label_scores, label_embeddings)

    return tf.reshape(averaged, [batch_size, batch_seq_len, label_embedding_dim])


dispatcher = {
  'label_attention': label_attention,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined value function `%s' % fn_name)
    exit(1)


def get_params(mode, value_map, train_outputs, features, labels, embeddings):
  params = {'mode': mode}
  params_map = value_map['params']
  for param_name, param_values in params_map.items():
    if 'label' in param_values:
      params[param_name] = labels[param_values['label']]
    elif 'embeddings' in param_values:
      params[param_name] = embeddings[param_values['feature']]
    elif 'feature' in param_values:
      params[param_name] = features[param_values['feature']]
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    elif 'layer' in param_values:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
  return params
