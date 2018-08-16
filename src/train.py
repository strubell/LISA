import tensorflow as tf
import argparse
import os
from functools import partial
import constants
import train_utils
import dataset
from vocab import Vocab
from model import LISAModel
import json
import numpy as np
import sys

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files',
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files',
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')

arg_parser.set_defaults(debug=False)

args, leftovers = arg_parser.parse_known_args()

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)

attention_configs = {}
if args.attention_configs and args.attention_configs != '':
  attention_config = train_utils.load_json_configs(args.attention_configs)

# Reverse the layer config because we really want a mapping from layer indices to task names,
# but json won't let us have integer keys :)
# layer_config = {v: k for k, v in layer_config.items()}

# Combine layer, task and layer, attention maps
layer_task_config = {}
layer_attention_config = {}
for task_or_attn_name, layer in layer_config.items():
  if task_or_attn_name in attention_config:
    layer_attention_config[layer] = attention_config[task_or_attn_name]
  elif task_or_attn_name in task_config:
    if layer not in layer_task_config:
      layer_task_config[layer] = {}
    layer_task_config[layer][task_or_attn_name] = task_config[task_or_attn_name]
  else:
    # todo make an error fn that does this
    tf.logging.log(tf.logging.ERROR, 'No task or attention config "%s"' % task_or_attn_name)
    sys.exit(1)

print("layer task config", layer_task_config)

print("layer attn config", layer_attention_config)

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.log(tf.logging.INFO, "Using TensorFlow version %s" % tf.__version__)

# Create a HParams object specifying the names and values of the model hyperparameters
# todo print or save hparams
hparams = tf.contrib.training.HParams(**constants.hparams)

# First get default hyperparams from the model config
if 'hparams' in model_config:
  # todo don't dump once this is actually json
  hparams_json = json.dumps(model_config['hparams'])
  hparams.parse_json(hparams_json)

# Override those with command line hyperparams
if args.hparams:
  hparams.parse(args.hparams)

# todo make these hparams
shuffle_buffer_multiplier = 100
eval_throttle_secs = 1000
eval_every_steps = 1000
if args.debug:
  shuffle_buffer_multiplier = 10
  eval_throttle_secs = 60
  eval_every_steps = 100
tf.logging.log(tf.logging.INFO, "Evaluating every %d steps" % eval_every_steps)

# Set the random seed. This defaults to int(time.time()) if not otherwise set.
np.random.seed(hparams.random_seed)
tf.set_random_seed(hparams.random_seed)

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

vocab = Vocab(args.train_file, data_config, args.save_dir)
vocab.update(args.dev_file)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]


def get_input_fn(data_file, num_epochs, shuffle, embedding_files):
  # this needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)

  return dataset.get_data_iterator(data_file, data_config, vocab_lookup_ops, hparams.batch_size, num_epochs, shuffle,
                                   shuffle_buffer_multiplier)


def train_input_fn():
  return get_input_fn(args.train_file, num_epochs=hparams.num_train_epochs, shuffle=True,
                      embedding_files=embedding_files)


def dev_input_fn():
  return get_input_fn(args.dev_file, num_epochs=1, shuffle=False, embedding_files=embedding_files)


# Generate mappings from feature/label names to indices in the model_fn inputs
feature_idx_map = {}
label_idx_map = {}
for i, f in enumerate([d for d in data_config.keys() if
                       ('feature' in data_config[d] and data_config[d]['feature']) or
                       ('label' in data_config[d] and data_config[d]['label'])]):
  if 'feature' in data_config[f] and data_config[f]['feature']:
    feature_idx_map[f] = i
  if 'label' in data_config[f] and data_config[f]['label']:
    if 'type' in data_config[f] and data_config[f]['type'] == 'range':
      idx = data_config[f]['conll_idx']
      j = i + idx[1] if idx[1] != -1 else -1
      label_idx_map[f] = (i, j)
    else:
      label_idx_map[f] = (i, i+1)


# Initialize the model
model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
                  vocab)

# Set up the Estimator
checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=eval_every_steps, keep_checkpoint_max=1)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)

# Set up early stopping -- always keep the model with the best F1
# todo: don't keep 5
save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn,
                                                                  key=task_config['best_eval_key']),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn)

# Train forever until killed
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=eval_throttle_secs,
                                  exporters=[save_best_exporter])

# Run training
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
