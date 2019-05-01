import tensorflow as tf
import json
import re
import sys
import dataset
import constants
from pathlib import Path


def load_hparams(args, model_config):
  # Create a HParams object specifying the names and values of the model hyperparameters
  hparams = tf.contrib.training.HParams(**constants.hparams)

  # First get default hyperparams from the model config
  if 'hparams' in model_config:
    hparams.override_from_dict(model_config['hparams'])

  if args.debug:
    hparams.set_hparam('shuffle_buffer_multiplier', 10)
    hparams.set_hparam('eval_throttle_secs', 60)
    hparams.set_hparam('eval_every_steps', 100)

  # Override those with command line hyperparams
  if args.hparams:
    hparams.parse(args.hparams)

  tf.logging.info("Using hyperparameters: %s" % str(hparams.values()))

  return hparams


# def get_input_fn(vocab, data_config, data_files, batch_size, num_epochs, shuffle, shuffle_buffer_multiplier=1):
#   # this needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
#   vocab_lookup_ops = vocab.create_vocab_lookup_ops()
#
#   return dataset.get_data_iterator(data_files, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
#                                    shuffle_buffer_multiplier)


def load_json_configs(config_file_list, args=None):
  """
  Loads a list of json configuration files into one combined map. Configuration files
  at the end of the list take precedence over earlier configuration files (so they will
  overwrite earlier configs!)

  If args is passed, then this function will attempt to replace entries surrounded with
  the special tokens ## ## with an entry from args with the same name.

  :param config_file_list: list of json configuration files to load
  :param args: command line args to replace special strings in json
  :return: map containing combined configurations
  """
  combined_config = {}
  if config_file_list:
    config_files = config_file_list.split(',')
    for config_file in config_files:
      if args:
        # read the json in as a string so that we can run a replace on it
        json_str = Path(config_file).read_text()
        matches = re.findall(r'.*##(.*)##.*', json_str)
        for match in matches:
          try:
            value = getattr(args, match)
            json_str = json_str.replace('##%s##' % match, value)
          except AttributeError:
            tf.logging.error('Could not find "%s" attribute in command line args when parsing: %s' %
                           (match, config_file))
            sys.exit(1)
        try:
          config = json.loads(json_str)
        except json.decoder.JSONDecodeError as e:
          tf.logging.error('Error reading json: "%s"' % config_file)
          tf.logging.error(e.msg)
          sys.exit(1)
      else:
        with open(config_file) as f:
          try:
            config = json.load(f)
          except json.decoder.JSONDecodeError as e:
            tf.logging.error('Error reading json: "%s"' % config_file)
            tf.logging.error(e.msg)
            sys.exit(1)
      combined_config = {**combined_config, **config}
  return combined_config


def copy_without_dropout(hparams):
  new_hparams = {k: (1.0 if 'dropout' in k else v) for k, v in hparams.values().items()}
  return tf.contrib.training.HParams(**new_hparams)


def get_vars_for_moving_average(average_norms):
  vars_to_average = tf.trainable_variables()
  if not average_norms:
    vars_to_average = [v for v in tf.trainable_variables() if 'norm' not in v.name]
  tf.logging.info("Creating moving averages for %d variables." % len(vars_to_average))
  return vars_to_average


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


def get_input_shapes(data_config, key):
  shapes = {}
  for _, config in data_config.items():
    for d, datum_config in config['mappings'].items():
      if key in datum_config and datum_config[key]:
        last_dims_shape = []
        if 'shape' in datum_config:
          last_dims_shape = [s if s else None for s in datum_config['shape']]
        # batch x sequence length x last_dims_shape...
        shapes[d] = [None, None] + last_dims_shape

  return shapes


def get_serving_input_receiver_fn(data_config):

  input_shapes = get_input_shapes(data_config, 'feature')
  print("input shapes", input_shapes)

  inputs = {k: tf.placeholder(tf.int64, shape) for k, shape in input_shapes.items()}

  def serving_input_receiver_fn():
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

  return serving_input_receiver_fn


# def serving_input_receiver_fn():
#   inputs = tf.placeholder(tf.int32, [None, None, None])
#   return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)
#
# def serving_input_receiver_fn():
#   inputs = {
#     "features": tf.placeholder(tf.int32, [None, None, None]),
#     "sentences": tf.placeholder(tf.int32, [None, None])
#   }
#   return tf.estimator.export.ServingInputReceiver(inputs, inputs)

