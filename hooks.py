import tensorflow as tf


class ValidationHook(tf.train.SessionRunHook):
  def __init__(self, estimator, input_fn, every_n_secs=None, every_n_steps=None):
    self._iter_count = 0
    self._estimator = estimator
    self._input_fn = input_fn
    self._timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
    self._should_trigger = False

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def before_run(self, run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

  def after_run(self, run_context, run_values):
    if self._should_trigger:
      self._estimator.evaluate(
        self._input_fn
      )
      self._timer.update_last_triggered_step(self._iter_count)
    self._iter_count += 1