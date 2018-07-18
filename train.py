import tensorflow as tf
import argparse
import dataset
from vocab import Vocab
import os
from LISA_model import LISAModel
from functools import partial
import train_utils

def get_input_fn(data_file, num_epochs, is_train):
  # this needs to be created from here so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = train_vocab.create_vocab_lookup_ops(args.word_embedding_file) if args.word_embedding_file \
    else train_vocab.create_vocab_lookup_ops()

  return dataset.get_data_iterator(data_file, data_config, vocab_lookup_ops, batch_size, num_epochs, is_train)


def train_input_fn():
  return get_input_fn(args.train_file, num_epochs=1, is_train=True)


def dev_input_fn():
  return get_input_fn(args.dev_file, num_epochs=1, is_train=False)


arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_file', type=str, help='Training data file')
arg_parser.add_argument('--dev_file', type=str, help='Development data file')
arg_parser.add_argument('--save_dir', type=str, help='Training data file')
arg_parser.add_argument('--word_embedding_file', type=str, help='File containing pre-trained word embeddings')

args = arg_parser.parse_args()

data_config = {
      'id': {
        'conll_idx': 0,
      },
      'word': {
        'conll_idx': 3,
        'feature': True,
        'vocab': 'glove.6B.100d.txt',
        'converter': 'lowercase',
        'oov': True
      },
      'auto_pos': {
        'conll_idx': 4,
        'vocab': 'gold_pos'
      },
      'gold_pos': {
        'conll_idx': 5,
        'label': True,
        'vocab': 'gold_pos'
      },
      'parse_head': {
        'conll_idx': 6,
        'label': True,
        'converter': 'parse_roots_self_loop'
      },
      'parse_label': {
        'conll_idx': 7,
        'label': True,
        'vocab': 'parse_label'
      },
      'domain': {
        'conll_idx': 0,
        'vocab': 'domain',
        'converter': 'strip_conll12_domain'
      },
      'predicate': {
        'conll_idx': 10,
        'label': True,
        'vocab': 'predicate',
        'converter': 'conll12_binary_predicates'
      },
      'srl': {
        'conll_idx': [14, -1],
        'label': True,
        'vocab': 'srl'
      },
    }

model_config = {
  'layers'
}

num_train_epochs = 50
batch_size = 256

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.logging.set_verbosity(tf.logging.INFO)

train_vocab = Vocab(args.train_file, data_config, args.save_dir)





model = LISAModel(args)

num_train_examples = 39832  # todo: compute this automatically
evaluate_every_n_epochs = 5
num_steps_in_epoch = int(num_train_examples / batch_size)
eval_every_steps = evaluate_every_n_epochs * num_steps_in_epoch
tf.logging.log(tf.logging.INFO, "Evaluating every %d steps" % eval_every_steps)

checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=eval_every_steps, keep_checkpoint_max=1)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)

# validation_hook = ValidationHook(estimator, dev_input_fn, every_n_steps=save_and_eval_every)

save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn, key="acc"),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_steps_in_epoch*num_train_epochs)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, exporters=[save_best_exporter])

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# estimator.train(input_fn=train_input_fn, steps=100000, hooks=[validation_hook])
# estimator.evaluate(input_fn=train_input_fn)


# np.set_printoptions(threshold=np.inf)
# with tf.Session() as sess:
#   sess.run(tf.tables_initializer())
#   for i in range(3):
#     print(sess.run(input_fn()))

