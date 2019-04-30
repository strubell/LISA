import tensorflow as tf
import argparse
import os
from functools import partial
import train_utils
from vocab import Vocab
from model import LISAModel
import numpy as np
import util

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files', required=True,
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files', required=True,
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
arg_parser.add_argument('--data_configs', required=True,
                        help='Comma-separated list of paths to data configuration json')
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')
arg_parser.add_argument('--num_gpus', type=int,
                        help='Number of GPUs for distributed training.')
arg_parser.add_argument('--keep_k_best_models', type=int,
                        help='Number of best models to keep.')
arg_parser.add_argument('--best_eval_key', required=True, type=str,
                        help='Key corresponding to the evaluation to be used for determining early stopping.')
arg_parser.add_argument('--use_xla', dest='use_xla', action='store_true',
                        help="Whether to use TensorFlow's XLA JIT.")

arg_parser.set_defaults(debug=False, num_gpus=1, keep_k_best_models=1, use_xla=False)

args, leftovers = arg_parser.parse_known_args()

util.init_logging(tf.logging.debug if args.debug else tf.logging.INFO)

# mixed precision training, via: https://medium.com/future-vision/bert-meets-gpus-403d3fbed848
# https://docs.nvidia.com/deeplearning/dgx/tensorflow-user-guide/index.html#tfamp
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Load all the various configurations
# todo: validate json
data_configs = [train_utils.load_json_configs(c) for c in args.data_configs.split(',')]
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)

# attention_config = {}
# if args.attention_configs and args.attention_configs != '':
#   attention_config = train_utils.load_json_configs(args.attention_configs)

# Combine layer, task and layer, attention maps
# todo save these maps in save_dir
layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config)

# Set the random seed. This defaults to int(time.time()) if not otherwise set.
np.random.seed(hparams.random_seed)
tf.set_random_seed(hparams.random_seed)

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

train_filenames = args.train_files.split(',')
dev_filenames = args.dev_files.split(',')

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map] + \
                  ["%s/vocab.txt" % embeddings_map['bert_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'bert_embeddings' in embeddings_map]

vocab = Vocab(data_configs, args.save_dir, train_filenames, embedding_files)
vocab.update(filenames=dev_filenames)

# todo: actually implement for multiple data configs!
# todo: each data config needs an associated data file
data_config = data_configs[0]


def train_input_fn():
  return train_utils.get_input_fn(vocab, data_config, train_filenames, hparams.batch_size,
                                  num_epochs=hparams.num_train_epochs, shuffle=False,
                                  shuffle_buffer_multiplier=hparams.shuffle_buffer_multiplier)


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False)


# Generate mappings from feature/label names to indices in the model_fn inputs
# feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)
# todo: don't hardcode!!
# feature_idx_map = util.load_input_idx_maps(data_config['mappings'], 'feature', ['feature'])
# label_idx_map = util.load_input_idx_maps(data_config['mappings'], 'label', ['label'])


# Initialize the model
model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, vocab)

if args.debug:
  tf.logging.info("Created trainable variables: %s" % str([v.name for v in tf.trainable_variables()]))

# Distributed training
distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus) if args.num_gpus > 1 else None

# Enable XLA JIT (via: https://medium.com/future-vision/bert-meets-gpus-403d3fbed848)
session_config = tf.ConfigProto()
# todo this doesn't currently work (on blake)
if args.use_xla:
  optimizer_options = session_config.graph_options.optimizer_options
  optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

# Set up the Estimator

# Checkpointing and XLA configuration
run_config = tf.estimator.RunConfig(save_checkpoints_steps=hparams.eval_every_steps, keep_checkpoint_max=1,
                                    train_distribute=distribution, session_config=session_config)

estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=run_config)

# Set up early stopping -- always keep the model with the best F1
export_assets = {"%s.txt" % vocab_name: "%s/assets.extra/%s.txt" % (args.save_dir, vocab_name)
                 for vocab_name in vocab.vocab_names_sizes.keys()}
tf.logging.info("Exporting assets: %s" % str(export_assets))
save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn,
                                                                  key=args.best_eval_key),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn,
                                               assets_extra=export_assets,
                                               exports_to_keep=args.keep_k_best_models)

# Train forever until killed
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=hparams.eval_throttle_secs,
                                  exporters=[save_best_exporter])

# Run training
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
