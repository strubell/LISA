import tensorflow as tf


def accuracy(predictions, targets, tokens_to_keep):
  return tf.metrics.accuracy(targets, predictions, weights=tokens_to_keep)


def srl_eval(predictions, targets, tokens_to_keep):

  # need to do embedding_lookup int->string
  # pass maps through

  # write file:
  #


  return tf.metrics.accuracy(targets, predictions)


dispatcher = {
  'accuracy': accuracy,
  'srl_eval': srl_eval,
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)


def get_params(predictions, task_map, train_outputs, task_labels, tokens_to_keep):
  params = {'predictions': predictions, 'targets': task_labels, 'tokens_to_keep': tokens_to_keep}
  if 'params' in task_map:
    params_map = task_map['params']
    for param_name, param_values in params_map.items():
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
  return params