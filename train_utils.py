import tensorflow as tf


def best_model_compare_fn(best_eval_result, current_eval_result, key):
  """Compares two evaluation results and returns true if the second one is greater.
    Both evaluation results should have the value for key, used for comparison.
    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.
      key: key to value used for comparison.
    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
      ValueError: If input eval result is None or no loss is available.
    """

  if not best_eval_result or key not in best_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  if not current_eval_result or key not in current_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  return best_eval_result[key] < current_eval_result[key]


def serving_input_receiver_fn():
  inputs = tf.placeholder(tf.int32, [None, None, None])
  return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)

