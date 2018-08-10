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

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_file', type=str, help='Training data file')
arg_parser.add_argument('--dev_file', type=str, help='Development data file')
arg_parser.add_argument('--save_dir', type=str, help='Training data file')
arg_parser.add_argument('--transition_stats', type=str, help='Transition statistics between labels')
arg_parser.add_argument('--bucket_boundaries', type=str, default='', help='Bucket boundaries for batching.')
arg_parser.add_argument('--hparams', type=str, default='', help='Comma separated list of "name=value" pairs.')
arg_parser.add_argument('--debug', dest='debug', action='store_true')
arg_parser.set_defaults(debug=False)

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
  'gold_pos': {
    'conll_idx': 4,
    'label': True,
    'vocab': 'gold_pos'
  },
  'auto_pos': {
    'conll_idx': 5,
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
    'conll_idx': [4, 9],
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
  'class_mlp_size': 100,
  'attn_mlp_size': 500,
  'hparams': {
    'label_smoothing': 0.1,
    'input_dropout': 0.8,
    'mlp_dropout': 0.9,
    'bilinear_dropout': 0.9,
    'attn_dropout': 0.9,
    'ff_dropout': 0.9,
    'prepost_dropout': 0.8,
    'moving_average_decay': 0.9999,
    'gradient_clip_norm': 5.0,
    'learning_rate': 0.04,
    'decay_rate': 1.5,
    'warmup_steps': 8000,
    'beta1': 0.9,
    'beta2': 0.98,
    'epsilon': 1e-12,
    'use_nesterov': True,
    'batch_size': 256
  },
  'layers': {
    'type': 'transformer',
    'num_heads': 8,
    'head_dim': 25,
    'ff_hidden_size': 800,
  },
  # todo add label embeddings for value fns here
  # todo also need to make it so that these can be grabbed as inputs to functions
  'embeddings': {
    'word_type': {
      'embedding_dim': 100,
      'pretrained_embeddings': 'embeddings/glove.6B.100d.txt'
    },
    'gold_pos': {
      'embedding_dim': 25,
    },
    'parse_label': {
      'embedding_dim': 25,
    },
    # 'predicate': {
    #   'embedding_dim': 100
    # }
  },
  'inputs': [
    'word_type',
    # 'predicate'
  ],
}

# todo validate these files
task_config = {
  'best_eval_key': 'srl_f1',
  'layers': {
    2: {
      'joint_pos_predicate': {
        'penalty': 1.0,
        'output_fn': {
          'name': 'joint_softmax_classifier',
          'params': {
            'joint_maps': {
              'joint_maps': [
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

    4: {
      'parse_head': {
        'penalty': 1.0,
        'output_fn': {
          'name': 'parse_bilinear',
          'params': {
          }
        },
        'eval_fns': {
          'label_accuracy': {
            'name': 'accuracy'
          }
        }
      },
      'parse_label': {
        'penalty': 0.1,
        'output_fn': {
          'name': 'conditional_bilinear',
          'params': {
            'dep_rel_mlp': {
              'layer': 'parse_head',
              'output': 'dep_rel_mlp'
            },
            'head_rel_mlp': {
              'layer': 'parse_head',
              'output': 'head_rel_mlp'
            },
            'parse_preds_train': {
              'label': 'parse_head'
            },
            'parse_preds_eval': {
              'layer': 'parse_head',
              'output': 'predictions'
            },
          }
        },
        'eval_fns': {
          'parse_eval': {
            'name': 'conll_parse_eval',
            'params': {
              'gold_parse_eval_file': {
                'value': args.save_dir + '/parse_gold.txt'
              },
              'pred_parse_eval_file': {
                'value': args.save_dir + '/parse_preds.txt'
              },
              'reverse_maps': {
                'reverse_maps': [
                  'word',
                  'parse_label',
                  'gold_pos'
                ]
              },
              'parse_head_predictions': {
                'layer': 'parse_head',
                'output': 'predictions'
              },
              'parse_head_targets': {
                'label': 'parse_head',
              },
              'words': {
                'feature': 'word',
              },
              'pos_targets': {
                'label': 'gold_pos'
              }
            }
          }
        }
      }
    },

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
                'reverse_maps': [
                  'word',
                  'srl',
                  'gold_pos'
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
              },
              'pos_predictions': {
                'layer': 'joint_pos_predicate',
                'output': 'gold_pos_predictions'
              },
              'pos_targets': {
                'label': 'gold_pos'
              }
            }
          }
        }
      }
    }
  }
}

attention_config = {
  3: {
    'value_fns': {
      'pos': {
        'name': 'label_attention',
        'params': {
          'train_label_scores': {
            'label': 'gold_pos'
          },
          'eval_label_scores': {
            'layer': 'joint_pos_predicate',
            'output': 'gold_pos_probabilities'
          },
          'label_embeddings': {
            'embeddings': 'gold_pos'
          }
        }
      }
    }
  },
  5: {
    'attention_fns': {
      'parse_heads': {
        'name': 'copy_from_predicted',
        'params': {
          'train_attention_to_copy': {
            'label': 'parse_head'
          },
          'eval_attention_to_copy': {
            'layer': 'parse_head',
            'output': 'scores'
          }
        }
      }
    },
    'value_fns': {
      'parse_label': {
        'name': 'label_attention',
        'params': {
          'train_labels': {
            'label': 'parse_label'
          },
          'eval_labels': {
            'layer': 'parse_label',
            'output': 'probabilities'
          },
          'label_embeddings': {
            'embeddings': 'parse_label'
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

vocab = Vocab(args.train_file, data_config, args.save_dir)
vocab.update(args.dev_file)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]

shuffle_buffer_multiplier = 100
eval_throttle_secs = 1000
eval_every_steps = 1000
num_train_examples = 39832  # todo: compute this automatically
# evaluate_every_n_epochs = 100
num_steps_in_epoch = int(num_train_examples / hparams.batch_size)
if args.debug:
  shuffle_buffer_multiplier = 10
  eval_throttle_secs = 60
  eval_every_steps = 100
tf.logging.log(tf.logging.INFO, "Evaluating every %d steps" % eval_every_steps)


# bucket_boundaries, test_bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES, constants.DEFAULT_BUCKET_BOUNDARIES
# if args.bucket_boundaries != '':
#   bucket_boundaries = np.loadtxt(args.bucket_boundaries, dtype=np.int32)
#   test_bucket_boundaries = [9, 13, 17, 20, 24, 28, 33, 40, 49, 116]


def get_input_fn(data_file, num_epochs, is_train, embedding_files):
  # this needs to be created from here so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)

  return dataset.get_data_iterator(data_file, data_config, vocab_lookup_ops, hparams.batch_size, num_epochs, is_train,
                                   shuffle_buffer_multiplier)


def train_input_fn():
  return get_input_fn(args.train_file, num_epochs=hparams.num_train_epochs, is_train=True,
                      embedding_files=embedding_files)


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


model = LISAModel(hparams, model_config, task_config['layers'], attention_config, feature_idx_map, label_idx_map, vocab)

checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=eval_every_steps, keep_checkpoint_max=1)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)

# validation_hook = train_hooks.ValidationHook(estimator, dev_input_fn, every_n_steps=eval_every_steps*1000)

save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn,
                                                                  key=task_config['best_eval_key']),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn) #max_steps=num_steps_in_epoch*num_train_epochs)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=eval_throttle_secs,
                                  exporters=[save_best_exporter])

# eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=1000, exporters=[save_best_exporter])
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# estimator.train(input_fn=train_input_fn, steps=100000, hooks=[validation_hook])
# estimator.evaluate(input_fn=train_input_fn)


# np.set_printoptions(threshold=np.inf)
# with tf.Session() as sess:
#   sess.run(tf.tables_initializer())
#   for i in range(3):
#     print(sess.run(train_input_fn()))

