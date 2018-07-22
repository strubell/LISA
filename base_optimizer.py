#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import constants

# ***************************************************************
class BaseOptimizer:
  """"""

  # =============================================================
  def __init__(self, *args, **kwargs):
    """"""

    self._global_step = kwargs.pop('global_step')
    self._accumulators = {}

    self.initial_learning_rate = kwargs.pop('learning_rate', constants.DEFAULT_LEARNING_RATE)
    self.decay_rate = kwargs.pop('decay_rate', constants.DEFAULT_DECAY_RATE)
    self.decay_steps = kwargs.pop('decay_steps', constants.DEFAULT_DECAY_STEPS)
    self.warmup_steps = kwargs.pop('warmup_steps', constants.DEFAULT_WARMUP_STEPS)
    self.gradient_clip_norm = kwargs.pop('gradient_clip_norm', constants.DEFAULT_GRADIENT_CLIP_NORM)
    self.chi = kwargs.pop('chi', constants.DEFAULT_CHI)




  # =============================================================
  def minimize(self, loss, name=None):
    """"""

    # Error checking
    var_list = tf.trainable_variables()
    for x_tm1 in var_list:
      if not isinstance(x_tm1, tf.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % x_tm1)
    if not var_list:
      raise ValueError("No variables to optimize")
    if loss.dtype.base_dtype != tf.float32:
      raise ValueError('Loss is not float32')

    # Compute gradients
    # var_refs = [x_tm1.read_value() for x_tm1 in var_list]
    grads = tf.gradients(loss, var_list,
                         colocate_gradients_with_ops=True,
                         gate_gradients=True,
                         aggregation_method=2)
    for x_tm1, g_t in zip(var_list, grads):
      if g_t is not None:
        if x_tm1.dtype.base_dtype != tf.float32:
          raise ValueError('%s is not float32' % x_tm1.name)

    # Apply gradients
    with tf.control_dependencies(None):
      self._init_acc(var_list, grads)
    with tf.name_scope(values=[], name=name, default_name=self.name) as name:
      caches = filter(lambda cache: cache['g_t'] is not None, self._prepare(var_list, grads))
      for cache in caches:
        x_tm1, g_t = cache['x_tm1'], cache['g_t']
        with tf.name_scope("update_" + x_tm1.op.name), tf.device(x_tm1.device):
          if isinstance(g_t, tf.Tensor):
            cache['g_t'] = tf.where(tf.is_finite(g_t), g_t, tf.zeros_like(g_t))
            self._apply_dense(cache)
          else:
            cache['g_t'] = tf.where(tf.is_finite(g_t.values), g_t.values, tf.zeros_like(g_t.values))
            cache['idxs'] = g_t.indices
            self._apply_sparse(cache)
      with tf.control_dependencies([self._finish(caches)]):
        with tf.device(self.global_step.device):
          return tf.assign_add(self.global_step, 1, name=name).op

  # =============================================================
  def _init_acc(self, var_list, grads):
    """"""

    for x_tm1, g_t in zip(var_list, grads):
      if self.chi > 0:
        tf.add_to_collection(self.get_accumulator(x_tm1, 'x'),
                             tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
        shape = self.get_variable_shape(x_tm1)
        if isinstance(g_t, tf.Tensor):
          self.get_accumulator(x_tm1, 'x/tm1', [])
        else:
          self.get_accumulator(x_tm1, 'x/tm1', [shape[0]] + [1] * (len(shape) - 1))
    return

  # =============================================================
  def _prepare(self, var_list, grads):
    """"""

    caches = []
    for x_tm1, g_t in zip(var_list, grads):
      caches.append({'x_tm1': x_tm1, 'g_t': g_t, 'updates': []})
    return caches

  # =============================================================
  def _apply_dense(self, cache):
    """"""

    raise NotImplementedError()

  # =============================================================
  def _apply_sparse(self, cache):
    """"""

    raise NotImplementedError()

  # =============================================================
  @staticmethod
  def get_variable_shape(x_tm1):
    return x_tm1.initialized_value().get_shape().as_list()

  # =============================================================
  def get_accumulator(self, x_tm1, acc_name, shape=None):
    """"""

    if shape is None:
      shape = self.get_variable_shape(x_tm1)

    if acc_name not in self._accumulators:
      self._accumulators[acc_name] = {}
    accumulator = self._accumulators[acc_name]

    if x_tm1 not in accumulator:
      new_name = '%s/%s' % (self.name, acc_name)
      zeros = tf.zeros(shape, dtype=x_tm1.dtype)
      with tf.name_scope('%s/%s' % (x_tm1.op.name, new_name)) as scope:
        with tf.device(x_tm1.device):
          accumulator[x_tm1] = b_tm1 = tf.Variable(zeros, name=scope, trainable=False)
          if isinstance(x_tm1, tf.Variable) and x_tm1._save_slice_info:
            real_acc_name = scope[len(x_tm1.op.name + '/'):-1]
            slice_info = x_tm1._save_slice_info
            b_tm1._set_save_slice_info(tf.Variable.SaveSliceInfo(
              '%s/%s' % (slice_info.full_name, real_slot_name),
              slice_info.full_shape[:],
              slice_info.var_offset[:],
              slice_info.var_shape[:]))
    return accumulator[x_tm1]

  # =============================================================
  def _dense_moving_average(self, x_tm1, a_t, name, beta=.9):
    """"""

    b_tm1 = self.get_accumulator(x_tm1, '%s' % name)
    tm1 = self.get_accumulator(x_tm1, '%s/tm1' % name, shape=[])
    t = tf.assign_add(tm1, 1)
    if beta < 1:
      beta_t = tf.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t = beta_t * (1 - beta ** tm1) / (1 - beta ** t)
    else:
      beta_t = tm1 / t
    b_t = tf.assign(b_tm1, beta_t * b_tm1)
    b_t = tf.assign_add(b_t, (1 - beta_t) * a_t)
    return b_t, t

  # =============================================================
  def _sparse_moving_average(self, x_tm1, idxs, a_t_, name, beta=.9):
    """"""

    b_tm1 = self.get_accumulator(x_tm1, '%s' % name)
    b_tm1_ = tf.gather(b_tm1, idxs)
    shape = self.get_variable_shape(x_tm1)
    tm1 = self.get_accumulator(x_tm1, '%s/tm1' % name, shape=[shape[0]] + [1] * (len(shape) - 1))
    tm1_ = tf.gather(tm1, idxs)
    t = tf.scatter_add(tm1, idxs, tf.ones_like(tm1_))
    t_ = tf.gather(t, idxs)
    if beta < 1:
      beta_t = tf.convert_to_tensor(beta, name='%s/decay' % name)
      beta_t_ = beta_t * (1 - beta_t ** tm1_) / (1 - beta_t ** t_)
    else:
      beta_t_ = tm1_ / t_
    b_t = tf.scatter_update(b_tm1, idxs, beta_t_ * b_tm1_)
    b_t = tf.scatter_add(b_t, idxs, (1 - beta_t_) * a_t_)
    return b_t, t

  # =============================================================
  def _finish(self, caches):
    """"""

    if self.gradient_clip_norm > 0:
      S_t = [cache['s_t'] for cache in caches]
      S_t, _ = tf.clip_by_global_norm(S_t, self.gradient_clip_norm)
      for cache, s_t in zip(caches, S_t):
        cache['s_t'] = s_t

    for cache in caches:
      x_tm1 = cache['x_tm1']
      s_t = cache['s_t']
      updates = cache['updates']
      with tf.name_scope('update_' + x_tm1.op.name), tf.device(x_tm1.device):
        if 'idxs' in cache:
          idxs = cache['idxs']
          x_t = tf.scatter_sub(x_tm1, idxs, s_t)
          if self.chi > 0:
            x_t_ = tf.gather(x_t, idxs)
            x_bar_t, t_x_bar = self._sparse_moving_average(x_tm1, idxs, x_t_, 'x', beta=self.chi)
        else:
          x_t = tf.assign_sub(x_tm1, s_t)
          if self.chi > 0:
            x_bar_t, t_x_bar = self._dense_moving_average(x_tm1, x_t, 'x', beta=self.chi)
      updates.append(x_t)
      if self.chi > 0:
        updates.extend([x_bar_t, t_x_bar])

    update_ops = [tf.group(*cache['updates']) for cache in caches]
    return tf.group(*update_ops, name='update')

  # ==============================================================
  def average(self, x_tm1):
    """"""

    if 'x' in self._accumulators:
      return x_tm1
      # return self._accumulators['x'].get(x_tm1, x_tm1)
    else:
      return x_tm1

  # ==============================================================
  def average_name(self, x_tm1):
    """"""

    return x_tm1.op.name + '/' + self.name + '/' + 'x'

  # ==============================================================
  def variables_to_restore(self, moving_avg_variables=None):
    """"""

    name_map = {}
    if moving_avg_variables is None:
      moving_avg_variables = tf.trainable_variables()
      moving_avg_variables += tf.moving_average_variables()
    # Remove duplicates
    moving_avg_variables = set(moving_avg_variables)
    # Collect all the variables with moving average,
    for v in moving_avg_variables:
      name_map[self.average_name(v)] = v
    # Make sure we restore variables without moving average as well.
    for v in list(set(tf.global_variables()) - moving_avg_variables):
      if v.op.name not in name_map:
        name_map[v.op.name] = v
    return name_map

  # ===============================================================
  @property
  def learning_rate(self):
    if self.warmup_steps > 0:
      lr = self.initial_learning_rate
      global_step_float = tf.cast(self.global_step, tf.float32)
      lr *= tf.minimum(tf.rsqrt(global_step_float), tf.multiply(global_step_float, self.warmup_steps ** -self.decay_rate))
      return lr
    else:
      if self.decay_steps > 0:
        return self.initial_learning_rate * self.decay_rate ** (self.global_step / self.decay_steps)
      else:
        return self.initial_learning_rate

  @property
  def global_step(self):
    return self._global_step

  @property
  def accumulators(self):
    return self._accumulators
