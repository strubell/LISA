
# __path__ = 'src/constants'

PAD_VALUE = -1
JOINT_LABEL_SEP = '/'

OOV_STRING = "<OOV>"

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
  'num_train_epochs': 100,
  'gradient_clip_norm': 5.0,
  'label_smoothing': 0.1,
  'moving_average_decay': 0.999,
  'input_dropout': 1.0,
  'bilin_keep_prob': 1.0
}


def get_default(name):
  try:
    return hparams[name]
  except KeyError:
    print('Undefined default hparam value `%s' % name)
    exit(1)
