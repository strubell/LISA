import tensorflow as tf


def copy_without_dropout(hparams):
  new_hparams = {k: (1.0 if 'dropout' in k else v) for k, v in hparams.values().items()}
  return tf.contrib.training.HParams(**new_hparams)


def learning_rate(hparams, global_step):
  lr = hparams.learning_rate
  warmup_steps = hparams.warmup_steps
  decay_rate = hparams.decay_rate
  if warmup_steps > 0:

    # add 1 to global_step so that we start at 1 instead of 0
    global_step_float = tf.cast(global_step, tf.float32) + 1.
    lr *= tf.minimum(tf.rsqrt(global_step_float),
                     tf.multiply(global_step_float, warmup_steps ** -decay_rate))
    return lr
  else:
    decay_steps = hparams.decay_steps
    if decay_steps > 0:
      return lr * decay_rate ** (global_step / decay_steps)
    else:
      return lr


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

