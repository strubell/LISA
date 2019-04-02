import tensorflow as tf
import numpy as np


def is_trainable(variable):
  return variable in tf.trainable_variables()


def get_num_trainable_parameters():
  return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])


def flip_gradient(x, lam=1.0):
  """Gradient reversal layer
     From: https://github.com/tachitachi/GradientReversal

     Implements the Gradient Reversal layer from Unsupervised Domain Adaptation by Backpropagation
     (https://arxiv.org/abs/1409.7495) and Domain-Adversarial Training of Neural Networks
     (https://arxiv.org/abs/1505.07818).

     The forward pass is the identify function, but the backward pass multiplies the gradients
     by -lam.

     x: Tensor input
     lam: gradient multiplier lambda
  """
  positive_path = tf.stop_gradient(x * tf.cast(1 + lam, tf.float32))
  negative_path = -x * tf.cast(lam, tf.float32)
  return positive_path + negative_path
