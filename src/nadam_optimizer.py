import tensorflow as tf
import constants


class NadamOptimizer(tf.contrib.optimizer_v2.OptimizerV2):

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, name="Adam"):
    super(NadamOptimizer, self).__init__(use_locking, name)

    self._set_hyper("learning_rate", learning_rate)
    self._set_hyper("beta1", beta1)
    self._set_hyper("beta2", beta2)
    self._set_hyper("epsilon", epsilon)

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

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).

    for x_tm1 in var_list:
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

  def _get_beta_accumulators(self, state=None):
    if state is None:
      state = self._get_per_graph_state()
    return (state.get_non_slot("beta1_power"),
            state.get_non_slot("beta2_power"))

  def _create_vars(self, var_list, state):
    # Non-slot variables end up on the same device(s).
    state.create_non_slot(initial_value=state.get_hyper("beta1"),
                          name="beta1_power")
    state.create_non_slot(initial_value=state.get_hyper("beta2"),
                          name="beta2_power")

    # Create slots for the first and second moments.
    for v in var_list:
      state.zeros_slot(v, "m")
      state.zeros_slot(v, "v")

  def _apply_dense(self, grad, var):
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

  def _apply_dense(self, grad, var, state):
    m = state.get_slot(var, "m")
    v = state.get_slot(var, "v")
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    return training_ops.apply_adam(
        var, m, v,
        math_ops.cast(beta1_power, var.dtype.base_dtype),
        math_ops.cast(beta2_power, var.dtype.base_dtype),
        state.get_hyper("learning_rate", var.dtype.base_dtype),
        state.get_hyper("beta1", var.dtype.base_dtype),
        state.get_hyper("beta2", var.dtype.base_dtype),
        state.get_hyper("epsilon", var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):

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

  def _apply_sparse_shared(self, grad, var, indices, scatter_add, state):
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
    beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
    lr_t = state.get_hyper("learning_rate", var.dtype.base_dtype)
    beta1_t = state.get_hyper("beta1", var.dtype.base_dtype)
    beta2_t = state.get_hyper("beta2", var.dtype.base_dtype)
    epsilon_t = state.get_hyper("epsilon", var.dtype.base_dtype)
    lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
    # m_t = beta1 * m + (1 - beta1) * g_t
    m = state.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = state_ops.assign(m, m * beta1_t,
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = scatter_add(m, indices, m_scaled_g_values)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = state.get_slot(var, "v")
    v_scaled_g_values = (grad * grad) * (1 - beta2_t)
    v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = scatter_add(v, indices, v_scaled_g_values)
    v_sqrt = math_ops.sqrt(v_t)
    var_update = state_ops.assign_sub(var,
                                      lr * m_t / (v_sqrt + epsilon_t),
                                      use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def _apply_sparse(self, grad, var, state):
    return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
            x, i, v, use_locking=self._use_locking),
        state)

  def _finish(self, state):
    # Update the power accumulators.
    beta1_power, beta2_power = self._get_beta_accumulators(state)
    update_beta1 = beta1_power.assign(
      beta1_power * state.get_hyper("beta1"),
      use_locking=self._use_locking)
    update_beta2 = beta2_power.assign(
      beta2_power * state.get_hyper("beta2"),
      use_locking=self._use_locking)
    return control_flow_ops.group(update_beta1, update_beta2)