import tensorflow as tf

"""Compares two evaluation results and returns true if the 2nd one is smaller.
  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.
  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.
  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.
  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
def best_model_compare_fn(best_eval_result, current_eval_result, key):

  if not best_eval_result or key not in best_eval_result:
    raise ValueError(
      'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or key not in current_eval_result:
    raise ValueError(
      'current_eval_result cannot be empty or no loss is found in it.')

  return best_eval_result[key] < current_eval_result[key]


def serving_input_receiver_fn():
  inputs = tf.placeholder(tf.int32, [None, None, None])
  return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)

