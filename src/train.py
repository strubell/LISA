import tensorflow as tf
import argparse
import os
from functools import partial
import constants
import train_utils
import dataset
import vocab
import model
import json

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_file', type=str, help='Training data file')
arg_parser.add_argument('--dev_file', type=str, help='Development data file')
arg_parser.add_argument('--save_dir', type=str, help='Training data file')
arg_parser.add_argument('--transition_stats', type=str, help='Transition statistics between labels')
arg_parser.add_argument('--hparams', type=str, default='', help='Comma separated list of "name=value" pairs.')

args = arg_parser.parse_args()

data_config = {
      'id': {
        'conll_idx': 2,
      },
      'sent_id': {
        'conll_idx': 1,
        'label': True
      },
      'word': {
        'conll_idx': 3,
        'feature': True,
        'vocab': 'word',
        'oov': False,
        'updatable': True
      },
      'word_type': {
        'conll_idx': 3,
        'feature': True,
        'vocab': 'embeddings/glove.6B.100d.txt',
        'converter':  {
          'name': 'lowercase'
        },
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
        'conll_idx': [6, 2],
        'label': True,
        'converter':  {
          'name': 'parse_roots_self_loop'
        }
      },
      'parse_label': {
        'conll_idx': 7,
        'label': True,
        'vocab': 'parse_label'
      },
      'domain': {
        'conll_idx': 0,
        'vocab': 'domain',
        'converter': {
          'name': 'strip_conll12_domain'
        }
      },
      'predicate': {
        'conll_idx': 9,
        'label': True,
        # 'feature': True,
        'vocab': 'predicate',
        'converter': {
          'name': 'conll12_binary_predicates'
        }
      },
      'joint_pos_predicate': {
        'conll_idx': [5, 9],
        'label': True,
        'vocab': 'joint_pos_predicate',
        'converter': {
          'name': 'joint_converter',
          'params': {
            'component_converters': ['default_converter', 'conll12_binary_predicates']
          }
        },
        'label_components': [
          'gold_pos',
          'predicate'
        ]
      },
      'srl': {
        'conll_idx': [14, -1],
        'type': 'range',
        'label': True,
        'vocab': 'srl',
        'converter': {
          'name': 'idx_range_converter'
        }
      },
    }

# todo define model inputs here
model_config = {
  'predicate_mlp_size': 200,
  'role_mlp_size': 200,
  'predicate_pred_mlp_size': 200,
  'hparams': {
    'label_smoothing': 0.1,
    'input_dropout': 0.8,
    'moving_average_decay': 0.999,
    'gradient_clip_norm': 5.0,
    'learning_rate': 0.0001, # 0.04,
    'decay_rate': 1.5,
    'warmup_steps': 8000,
    'beta1': 0.9,
    'beta2': 0.999, #0.98,
    'epsilon': 1e-12,
    'batch_size': 256
  },
  'layers': {
    'type': 'transformer',
    'num_heads': 8,
    'head_dim': 25,
    'ff_hidden_size': 800,
    'attn_dropout': 0.9,
    'ff_dropout': 0.9,
    'prepost_dropout': 0.9,
  },
  'inputs': {
    'word_type': {
      'embedding_dim': 100,
      'pretrained_embeddings': 'embeddings/glove.6B.100d.txt'
    },
    # 'predicate': {
    #   'embedding_dim': 100
    # }
  }
}

# task_config = {
#   'gold_pos': {
#     'layer': 3,
#   },
#   'predicate': {
#     'layer': 3,
#   },
#   'parse_head': {
#     'layer': 5,
#   },
#   'parse_label': {
#     'layer': 5,
#   },
#   'srl': {
#     'layer': 12
#   },
# }
# todo validate these files
task_config = {
  'best_eval_key': 'srl_f1',
  'layers': {
    3: {
      'joint_pos_predicate': {
        'penalty': 1.0,
        'output_fn': {
          'name': 'joint_softmax_classifier',
          'params': {
            'joint_maps': {
              'maps': [
                'joint_pos_predicate_to_gold_pos',
                'joint_pos_predicate_to_predicate'
              ]
            }
          }
        },
        'eval_fns': {
          'predicate_acc': {
            'name': 'accuracy',
            'params': {
              'predictions': {
                'layer': 'joint_pos_predicate',
                'output': 'predicate_predictions'
              },
              'targets': {
                'label': 'predicate'
              }
            }
          },
          'pos_acc': {
            'name': 'accuracy',
            'params': {
              'predictions': {
                'layer': 'joint_pos_predicate',
                'output': 'gold_pos_predictions'
              },
              'targets': {
                'label': 'gold_pos'
              }
            }
          }
        }
      }
    },

    # 5: {
    #   'parse_head',
    #   'parse_label'
    # },

    11: {
      'srl': {
        'penalty': 1.0,
        'viterbi': True,
        'transition_stats': args.transition_stats,
        'output_fn': {
          'name': 'srl_bilinear',
          'params': {
            'predicate_targets': {
              'label': 'predicate'
            },
            'predicate_preds_train': {
              'label': 'predicate'
            },
            'predicate_preds_eval': {
              'layer': 'joint_pos_predicate',
              'output': 'predicate_predictions'
            }
          }
        },
        'eval_fns': {
          'srl_f1': {
            'name': 'conll_srl_eval',
            'params': {
              'gold_srl_eval_file': {
                'value': args.save_dir + '/srl_gold.txt'
              },
              'pred_srl_eval_file': {
                'value': args.save_dir + '/srl_preds.txt'
              },
              'reverse_maps': {
                'maps': [
                  'word',
                  'srl'
                ]
              },
              'targets': {
                'layer': 'srl',
                'output': 'targets'
              },
              'predicate_targets': {
                'label': 'predicate',
              },
              'words': {
                'feature': 'word',
              },
              'predicate_predictions': {
                'layer': 'joint_pos_predicate',
                'output': 'predicate_predictions'
              }
            }
          }
        }
      }
    }
  }
}

# Create a HParams object specifying the names and values of the
# model hyperparameters:
hparams = tf.contrib.training.HParams(**constants.hparams)

# First get default hyperparams from the model config
if 'hparams' in model_config:
  # todo don't dump once this is actually json
  hparams_json = json.dumps(model_config['hparams'])
  hparams.parse_json(hparams_json)

# Override those with command line hyperparams
hparams.parse(args.hparams)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.logging.set_verbosity(tf.logging.INFO)

vocab = vocab.Vocab(args.train_file, data_config, args.save_dir)
vocab.update(args.dev_file)

embedding_files = [input_map['pretrained_embeddings'] for input_map in model_config['inputs'].values()
                   if 'pretrained_embeddings' in input_map]


def get_input_fn(data_file, num_epochs, is_train, embedding_files):
  # this needs to be created from here so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)

  return dataset.get_data_iterator(data_file, data_config, vocab_lookup_ops, hparams.batch_size, num_epochs, is_train)


def train_input_fn():
  return get_input_fn(args.train_file, num_epochs=hparams.num_train_epochs, is_train=True, embedding_files=embedding_files)


def dev_input_fn():
  return get_input_fn(args.dev_file, num_epochs=1, is_train=False, embedding_files=embedding_files)


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


model = model.LISAModel(hparams, model_config, task_config['layers'], feature_idx_map, label_idx_map, vocab)

num_train_examples = 39832  # todo: compute this automatically
evaluate_every_n_epochs = 100
num_steps_in_epoch = int(num_train_examples / hparams.batch_size)
eval_every_steps = 1000
tf.logging.log(tf.logging.INFO, "Evaluating every %d steps" % eval_every_steps)

checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=eval_every_steps, keep_checkpoint_max=1)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)

# validation_hook = train_hooks.ValidationHook(estimator, dev_input_fn, every_n_steps=eval_every_steps*1000)

save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn,
                                                                  key=task_config['best_eval_key']),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn) #max_steps=num_steps_in_epoch*num_train_epochs)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=600, exporters=[save_best_exporter])
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# estimator.train(input_fn=train_input_fn, steps=100000, hooks=[validation_hook])
# estimator.evaluate(input_fn=train_input_fn)


# np.set_printoptions(threshold=np.inf)
# with tf.Session() as sess:
#   sess.run(tf.tables_initializer())
#   for i in range(3):
#     print(sess.run(train_input_fn()))

