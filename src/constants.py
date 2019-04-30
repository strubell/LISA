import time
import tensorflow as tf

PAD_VALUE = -1
JOINT_LABEL_SEP = '/'

OOV_STRING = "[OOV]"

DEFAULT_BUCKET_BOUNDARIES = [20, 30, 50, 80]

VERY_LARGE = 1e9
VERY_SMALL = -1e9

BERT_PAD = '[PAD]'
BERT_CLS = '[CLS]'
BERT_SEP = '[SEP]'
BERT_MASK_STRS = [BERT_PAD, BERT_CLS, BERT_SEP]

# Optimizer hyperparameters
hparams = {
  'learning_rate': 0.04,
  'decay_rate': 1.5,
  'decay_steps': 5000,
  'warmup_steps': 8000,
  'beta1': 0.9,
  'beta2': 0.98,
  'epsilon': 1e-12,
  'use_nesterov': True,
  'batch_size': 256,
  'shuffle_buffer_multiplier': 100,
  'eval_throttle_secs': 1000,
  'eval_every_steps': 1000,
  'num_train_epochs': 10000,
  'gradient_clip_norm': 5.0,
  'label_smoothing': 0.1,
  'moving_average_decay': 0.999,
  'average_norms': False,
  'input_dropout': 1.0,
  'bilinear_dropout': 1.0,
  'mlp_dropout': 1.0,
  'attn_dropout': 1.0,
  'ff_dropout': 1.0,
  'prepost_dropout': 1.0,
  'random_seed': int(time.time())
}


def get_default(name):
  try:
    return hparams[name]
  except KeyError:
    print('Undefined default hparam value `%s' % name)
    exit(1)
