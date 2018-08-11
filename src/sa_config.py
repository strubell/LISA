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
          'train_label_scores': {
            'label': 'parse_label'
          },
          'eval_label_scores': {
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