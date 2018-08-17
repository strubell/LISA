import tensorflow as tf
import argparse
import train_utils
from vocab import Vocab
from model import LISAModel
import sys

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

# todo: are these necessary?
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

attention_config = {}
if args.attention_configs and args.attention_configs != '':
  attention_config = train_utils.load_json_configs(args.attention_configs)

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

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.log(tf.logging.INFO, "Using TensorFlow version %s" % tf.__version__)

hparams = train_utils.load_hparams(args, model_config)

dev_filenames = args.dev_files.split(',')
test_filenames = args.test_files.split(',') if args.test_files else []

vocab = Vocab(data_config, args.save_dir)
vocab.update(test_filenames)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]

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
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir)


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False,
                                  embedding_files=embedding_files)


estimator.evaluate(input_fn=dev_input_fn, checkpoint_path="%s/export/best_exporter" % args.save_dir)

for test_file in test_filenames:
  def test_input_fn():
    return train_utils.get_input_fn(vocab, data_config, test_file, hparams.batch_size, num_epochs=1, shuffle=False,
                                    embedding_files=embedding_files)

  estimator.evaluate(input_fn=test_input_fn, checkpoint_path="%s/export/best_exporter" % args.save_dir)

