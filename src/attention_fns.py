import tensorflow as tf
import constants


def copy_from_predicted(mode, train_attention_to_copy, eval_attention_to_copy):
  attention_to_copy = train_attention_to_copy if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_to_copy
  if len(attention_to_copy.get_shape()) < 3:
    # use non-standard on and off values because we're going to softmax this later, and
    # want the softmaxed result to be 0/1
    attention_to_copy = tf.one_hot(attention_to_copy, tf.shape(attention_to_copy)[-1], on_value=constants.VERY_LARGE,
                                   off_value=constants.VERY_SMALL)

  return tf.cast(attention_to_copy, tf.float32)


dispatcher = {
  'copy_from_predicted': copy_from_predicted,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined attention function `%s' % fn_name)
    exit(1)


def get_params(mode, attn_map, train_outputs, features, labels):
  params = {'mode': mode}
  params_map = attn_map['params']
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through
    if 'label' in param_values:
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
