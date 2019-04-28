import tensorflow as tf
import numpy as np
import collections
import re


def is_trainable(variable):
  return variable in tf.trainable_variables()


def get_num_parameters(variables):
  return np.sum([np.prod(v.shape) for v in variables])


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    current_scope = tf.get_variable_scope().name
    name_in_scope = "%s/%s" % (current_scope, name)
    if name_in_scope not in name_to_variable:
      continue
    assignment_map[name] = name_in_scope
    initialized_variable_names[name_in_scope] = 1
    initialized_variable_names[name_in_scope + ":0"] = 1

  return (assignment_map, initialized_variable_names)


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
