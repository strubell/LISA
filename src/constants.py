import time

PAD_VALUE = -1
JOINT_LABEL_SEP = '/'

OOV_STRING = "<OOV>"

DEFAULT_BUCKET_BOUNDARIES = [20, 30, 50, 80]

VERY_LARGE = 1e9
VERY_SMALL = -1e9

# Optimizer hyperparameters
hparams = {
  'learning_rate': 0.04,
  'decay_rate': 1.5,
  'decay_steps': 5000,
  'warmup_steps': 8000,
  'beta1': 0.9,
  'beta2': 0.98,
  'gamma': 0,
  'epsilon': 1e-12,
  'use_nesterov': True,
  'chi': 0,
  'batch_size': 256,
  'num_train_epochs': 10000,
  'gradient_clip_norm': 5.0,
  'label_smoothing': 0.1,
  'moving_average_decay': 0.999,
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
