import tensorflow as tf
import argparse
import dataset
from vocab import Vocab
import os
from LISA_model import LISAModel
from functools import partial

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_file', type=str, help='Training data file')
arg_parser.add_argument('--dev_file', type=str, help='Development data file')
arg_parser.add_argument('--save_dir', type=str, help='Training data file')
arg_parser.add_argument('--word_embedding_file', type=str, help='File containing pre-trained word embeddings')

args = arg_parser.parse_args()

data_config = {
      'id': {
        'idx': 0,
      },
      'word': {
        'idx': 3,
        'feature': True,
        'vocab': 'glove.6B.100d.txt',
        'converter': 'lowercase',
        'oov': True
      },
      'auto_pos': {
        'idx': 4,
        'vocab': 'gold_pos'
      },
      'gold_pos': {
        'idx': 5,
        'label': True,
        'vocab': 'gold_pos'
      },
      'parse_head': {
        'idx': 6,
        'label': True,
        'converter': 'parse_roots_self_loop'
      },
      'parse_label': {
        'idx': 7,
        'label': True,
        'vocab': 'parse_label'
      },
      'domain': {
        'idx': 0,
        'vocab': 'domain',
        'converter': 'strip_conll12_domain'
      },
      'predicate': {
        'idx': 10,
        'label': True,
        'vocab': 'predicate',
        'converter': 'conll12_binary_predicates'
      },
      'srl': {
        'idx': [14, -1],
        'label': True,
        'vocab': 'srl'
      },
    }

num_train_epochs = 50
batch_size = 256

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.logging.set_verbosity(tf.logging.INFO)

train_vocab = Vocab(args.train_file, data_config, args.save_dir)

def get_input_fn(data_file, num_epochs, is_train):
  # this needs to be created from here so that it ends up in the same tf.Graph as everything else
  # vocab_lookup_ops = train_vocab.get_lookup_ops(args.word_embedding_file) if args.word_embedding_file \
  #   else train_vocab.get_lookup_ops()
  vocab_lookup_ops = train_vocab.create_vocab_lookup_ops(args.word_embedding_file) if args.word_embedding_file \
    else train_vocab.create_vocab_lookup_ops()

  return dataset.get_data_iterator(data_file, data_config, vocab_lookup_ops, batch_size, num_epochs, is_train)


def train_input_fn():
  return get_input_fn(args.train_file, num_epochs=num_train_epochs, is_train=True)

def dev_input_fn():
  return get_input_fn(args.dev_file, num_epochs=1, is_train=False)


model = LISAModel(args)

num_train_examples = 39832  # todo: compute this automatically
num_steps_in_epoch = int(num_train_examples / batch_size)

checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=num_steps_in_epoch, keep_checkpoint_max=1)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)
# estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir)


# validation_hook = ValidationHook(estimator, dev_input_fn, every_n_steps=save_and_eval_every)




# def __init__(self,
#              name='best_exporter',
#              serving_input_receiver_fn=None,
#              event_file_pattern='eval/*.tfevents.*',
#              compare_fn=_loss_smaller,
#              assets_extra=None,
#              as_text=False,
#              exports_to_keep=5):


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

  return best_eval_result[key] > current_eval_result[key]


save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(best_model_compare_fn, key="acc"),
                                               serving_input_receiver_fn=tf.estimator.export.build_parsing_serving_input_receiver_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100000)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, steps=num_steps_in_epoch, exporters=[save_best_exporter])

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# estimator.train(input_fn=train_input_fn, steps=100000, hooks=[validation_hook])
#
# estimator.evaluate(input_fn=train_input_fn)

# np.set_printoptions(threshold=np.inf)
# with tf.Session() as sess:
#   sess.run(tf.tables_initializer())
#   for i in range(3):
#     print(sess.run(input_fn()))

