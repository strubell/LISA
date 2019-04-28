import tensorflow as tf
import argparse
import train_utils
import tf_utils
import os
from vocab import Vocab
from model import LISAModel
import util

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--test_files',
                        help='Comma-separated list of test data files')
arg_parser.add_argument('--dev_files',
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory containing saved model')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
# todo: are these necessary?
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')
arg_parser.add_argument('--combine_test_files', action='store_true',
                        help='Whether to combine list of test files into a single score.')

arg_parser.set_defaults(debug=False, combine_test_files=False)


args, leftovers = arg_parser.parse_known_args()

util.init_logging(tf.logging.debug if args.debug else tf.logging.INFO)

if not os.path.isdir(args.save_dir):
  util.fatal_error("save_dir not found: %s" % args.save_dir)

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)

layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config)

dev_filenames = args.dev_files.split(',')
test_filenames = args.test_files.split(',') if args.test_files else []

# todo: fix data configs, embeddings
vocab = Vocab(data_config, args.save_dir)
vocab.update(test_filenames)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]

# Generate mappings from feature/label names to indices in the model_fn inputs
feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

# Initialize the model
model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
                  vocab)
tf.logging.info("Created model with %d parameters" % tf_utils.get_num_parameters(tf.trainable_variables()))


# Set up the Estimator
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir)


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False)


tf.logging.info("Evaluating on dev files: %s" % str(dev_filenames))
estimator.evaluate(input_fn=dev_input_fn)

if args.combine_test_files:
  def test_input_fn():
    return train_utils.get_input_fn(vocab, data_config, test_filenames, hparams.batch_size, num_epochs=1, shuffle=False)

  tf.logging.info("Evaluating on test files: %s" % str(test_filenames))
  estimator.evaluate(input_fn=test_input_fn)

else:
  for test_file in test_filenames:
    def test_input_fn():
      return train_utils.get_input_fn(vocab, data_config, [test_file], hparams.batch_size, num_epochs=1, shuffle=False)


    tf.logging.info("Evaluating on test file: %s" % str(test_file))
    estimator.evaluate(input_fn=test_input_fn)

