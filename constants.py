
PAD_VALUE = -1
JOINT_LABEL_SEP = '/'

OOV_STRING = "<OOV>"


# DEFAULT_LEARNING_RATE = 0.04
# DEFAULT_DECAY_RATE = 1.5
# DEFAULT_DECAY_STEPS = 5000
# DEFAULT_WARMUP_STEPS = 8000
# DEFAULT_MU = 0.9
# DEFAULT_NU = 0.98
# DEFAULT_GAMMA = 0
# DEFAULT_EPSILON = 1e-12
# DEFAULT_CHI = 0
# DEFAULT_GRADIENT_CLIP_NORM = 1.0

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
  'chi': 0,
  'gradient_clip_norm': 5.0,
  'label_smoothing': 0.1,
  'moving_average_decay': 0.999,
  'input_dropout': 1.0,
  'bilin_keep_prob': 1.0
}


def get_default(name):
  return hparams[name]
