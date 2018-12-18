import tensorflow as tf
import numpy as np
import argparse
import train_utils
from vocab import Vocab
import sys
from tensorflow.contrib import predictor
import evaluation_fns as eval_fns
import constants


def sequence_mask_np(lengths, maxlen=None):
  if not maxlen:
    maxlen = np.max(lengths)
  return np.arange(maxlen) < np.array(lengths)[:, None]


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

arg_parser.add_argument('--ensemble', dest='ensemble', action='store_true',
                        help='Whether to ensemble models in save dir.')

arg_parser.set_defaults(debug=False, ensemble=False)

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
tf.logging.log(tf.logging.INFO, "Using Python version %s" % sys.version)
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

pred_srl_eval_file = task_config['srl']['eval_fns']['srl_f1']['params']['pred_srl_eval_file']['value']
gold_srl_eval_file = task_config['srl']['eval_fns']['srl_f1']['params']['gold_srl_eval_file']['value']

# Initialize the model
# model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
#                   vocab)
#
# # Set up the Estimator
# estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir)


predict_fn = predictor.from_saved_model(args.save_dir)


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False,
                                  embedding_files=embedding_files)


def eval_fn(input_fn):
  input_op = input_fn()
  srl_correct_total = srl_excess_total = srl_missed_total = 0.
  with tf.get_default_session():
    while True:
      try:
        # input_np = sess.run(dev_input_fn())
        input_np = input_op.eval()
        predictor_input = {'input': input_np}
        predictions = predict_fn(predictor_input)

        srl_predictions = predictions['srl_predictions']
        predicate_predictions = predictions['joint_pos_predicate_predicate_predictions']

        feats = {f: input_np[:, :, idx] for f, idx in feature_idx_map.items()}

        str_srl_predictions = [list(map(vocab.reverse_maps['srl'].get, s)) for s in srl_predictions]
        str_words = [list(map(vocab.reverse_maps['word'].get, s)) for s in feats['word']]

        tokens_to_keep = np.where(feats['word'] == constants.PAD_VALUE, 0, 1)

        labels = {}
        for l, idx in label_idx_map.items():
          these_labels = input_np[:, :, idx[0]:idx[1]] if idx[1] != -1 else input_np[:, :, idx[0]:]
          these_labels_masked = np.multiply(these_labels, np.expand_dims(tokens_to_keep, -1))
          # check if we need to mask another dimension
          if idx[1] == -1:
            this_mask = np.where(these_labels_masked == constants.PAD_VALUE, 0, 1)
            these_labels_masked = np.multiply(these_labels_masked, this_mask)
          else:
            these_labels_masked = np.squeeze(these_labels_masked, -1)
          labels[l] = these_labels_masked

        predicate_targets = labels['predicate']

        predicates_per_sent = np.sum(predicate_targets, axis=-1)
        predicates_indices = np.where(sequence_mask_np(predicates_per_sent))
        srl_targets = np.transpose(labels['srl'], [0, 2, 1])
        gathered_srl_targets = srl_targets[predicates_indices]
        str_srl_targets = [list(map(vocab.reverse_maps['srl'].get, s)) for s in gathered_srl_targets]

        srl_correct, srl_excess, srl_missed = eval_fns.conll_srl_eval_py(str_srl_predictions, predicate_predictions,
                                                                         str_words, tokens_to_keep, str_srl_targets,
                                                                         predicate_targets,
                                                                         pred_srl_eval_file, gold_srl_eval_file)

        srl_correct_total += srl_correct
        srl_excess_total += srl_excess
        srl_missed_total += srl_missed
      except tf.errors.OutOfRangeError:
        break

  precision = srl_correct_total / (srl_correct_total + srl_excess_total)
  recall = srl_correct_total / (srl_correct_total + srl_missed_total)
  f1 = 2 * precision * recall / (precision + recall)

  tf.logging.log(tf.logging.INFO,
                 "SRL precision: %2.2f; recall: %2.2f; F1: %2.2f" % (precision * 100, recall * 100, f1 * 100))


with tf.Session() as sess:
  sess.run(tf.tables_initializer())

tf.logging.log(tf.logging.INFO, "Evaluating on dev files: %s" % str(dev_filenames))
eval_fn(dev_input_fn)

for test_file in test_filenames:
  def test_input_fn():
    return train_utils.get_input_fn(vocab, data_config, [test_file], hparams.batch_size, num_epochs=1, shuffle=False,
                                    embedding_files=embedding_files)


  tf.logging.log(tf.logging.INFO, "Evaluating on test file: %s" % str(test_file))
  eval_fn(test_input_fn)

