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

from base_optimizer import BaseOptimizer


# ***************************************************************
class RadamOptimizer(BaseOptimizer):
  """"""

  def __init__(self, *args, **kwargs):
    super.__init__(*args, **kwargs)

    self._name = "RadamOptimizer"

    self.mu = kwargs.pop('mu', constants.DEFAULT_MU)
    self.nu = kwargs.pop('mu', constants.DEFAULT_MU)
    self.gamma = kwargs.pop('gamma', constants.DEFAULT_GAMMA)

    self.epsilon = kwargs.pop('epsilon', constants.DEFAULT_EPSILON)




  # =============================================================
  def _init_acc(self, var_list, grads):
    """"""

    super(RadamOptimizer, self)._init_acc(var_list, grads)
    for x_tm1, g_t in zip(var_list, grads):
      if self.mu > 0:
        self.get_accumulator(x_tm1, 'm')
        shape = self.get_variable_shape(x_tm1)
        if isinstance(g_t, tf.Tensor):
          self.get_accumulator(x_tm1, 'm/tm1', [])
        else:
          self.get_accumulator(x_tm1, 'm/tm1', [shape[0]] + [1] * (len(shape) - 1))
      if self.nu > 0:
        self.get_accumulator(x_tm1, 'v')
        shape = self.get_variable_shape(x_tm1)
        if isinstance(g_t, tf.Tensor):
          self.get_accumulator(x_tm1, 'v/tm1', [])
        else:
          self.get_accumulator(x_tm1, 'v/tm1', [shape[0]] + [1] * (len(shape) - 1))
    return

  # =============================================================
  def _apply_dense(self, cache):
    """"""

    x_tm1, g_t = cache['x_tm1'], cache['g_t']
    updates = cache['updates']

    if self.mu > 0:
      m_t, t_m = self._dense_moving_average(x_tm1, g_t, 'm', beta=self.mu)
      m_bar_t = (1 - self.gamma) * m_t + self.gamma * g_t
      updates.extend([m_t, t_m])
    else:
      m_bar_t = g_t

    if self.nu > 0:
      v_t, t_v = self._dense_moving_average(x_tm1, g_t ** 2, 'v', beta=self.nu)
      v_bar_t = tf.sqrt(v_t + self.epsilon)
      updates.extend([v_t, t_v])
    else:
      v_bar_t = 1

    s_t = self.learning_rate * m_bar_t / v_bar_t
    cache['s_t'] = s_t
    return cache

  # =============================================================
  def _apply_sparse(self, cache):
    """"""

    x_tm1, g_t, idxs = cache['x_tm1'], cache['g_t'], cache['idxs']
    idxs, idxs_ = tf.unique(idxs)
    g_t_ = tf.unsorted_segment_sum(g_t, idxs_, tf.size(idxs))
    updates = cache['updates']

    if self.mu > 0:
      m_t, t_m = self._sparse_moving_average(x_tm1, idxs, g_t_, 'm', beta=self.mu)
      m_t_ = tf.gather(m_t, idxs)
      m_bar_t_ = (1 - self.gamma) * m_t_ + self.gamma * g_t_
      updates.extend([m_t, t_m])
    else:
      m_bar_t_ = g_t_

    if self.nu > 0:
      v_t, t_v = self._sparse_moving_average(x_tm1, idxs, g_t_ ** 2, 'v', beta=self.nu)
      v_t_ = tf.gather(v_t, idxs)
      v_bar_t_ = tf.sqrt(v_t_ + self.epsilon)
      updates.extend([v_t, t_v])
    else:
      v_bar_t_ = 1

    s_t_ = self.learning_rate * m_bar_t_ / v_bar_t_
    cache['s_t'] = s_t_
    cache['g_t'] = g_t_
    cache['idxs'] = idxs
    return cache
