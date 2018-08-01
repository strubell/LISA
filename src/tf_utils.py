import tensorflow as tf


def is_trainable(variable):
  return variable in tf.trainable_variables()
