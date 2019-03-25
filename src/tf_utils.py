import tensorflow as tf
import numpy as np


def is_trainable(variable):
  return variable in tf.trainable_variables()


def get_num_trainable_parameters():
  return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
