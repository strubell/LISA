import tensorflow as tf


def accuracy(preds, targets, tokens_to_keep):
  return tf.metrics.accuracy(targets, preds, weights=tokens_to_keep)



dispatcher = {
  'accuracy': accuracy,
}

def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)

def get_params(predictions, task_map, train_outputs, task_labels, tokens_to_keep):
  params = {'predictions': predictions, 'targets': task_labels, 'tokens_to_keep': tokens_to_keep}
  params_map = task_map['params']
  for param_name, param_values in params_map.items():
    outputs_layer = train_outputs[param_values['layer']]
    params[param_name] = outputs_layer[param_values['output']]
  return params