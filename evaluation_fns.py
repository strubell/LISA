import tensorflow as tf
import os
from subprocess import check_output, CalledProcessError


def accuracy(predictions, targets, mask):
  return tf.metrics.accuracy(targets, predictions, weights=mask)


def conll_parse_eval(predictions, targets, mask, reverse_maps, gold_parse_eval_file, pred_parse_eval_file):
  # TODO write this function
  return tf.metrics.accuracy(targets, predictions, mask)


def conll_srl_eval(predictions, targets, mask, reverse_maps, gold_srl_eval_file, pred_srl_eval_file):

  # First write predictions file w/ format:
  # -        (A1*  (A1*
  # -          *     *
  # -          *)    *)
  # -          *     *
  # expected (V*)    *
  # -        (C-A1*  *
  # widen     *     (V*)
  # -         *     (A4*
  # with open(pred_srl_eval_file, 'w') as f:


  # srl_eval = check_output(["perl", "bin/srl-eval.pl", srl_gold_fname, srl_preds_fname], stderr=devnull)
  # print(srl_eval)
  # overall_f1 = float(srl_eval.split('\n')[6].split()[-1])

  return tf.metrics.accuracy(targets, predictions, mask)


dispatcher = {
  'accuracy': accuracy,
  'conll_srl_eval': conll_srl_eval,
  'conll_parse_eval': conll_parse_eval,

}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)


def get_params(task_outputs, task_map, train_outputs, task_labels, lookup_maps, tokens_to_keep):
  targets = task_labels if 'targets' not in task_outputs else task_outputs['targets']
  mask = tokens_to_keep if 'mask' not in task_outputs else task_outputs['mask']
  params = {'predictions': task_outputs['predictions'], 'targets': targets, 'mask': mask}
  if 'params' in task_map:
    params_map = task_map['params']
    for param_name, param_values in params_map.items():
      if 'maps' in param_values:
        params[param_name] = {map_name: lookup_maps[map_name] for map_name in param_values['maps']}
      elif 'layer' in param_values:
        outputs_layer = train_outputs[param_values['layer']]
        params[param_name] = outputs_layer[param_values['output']]
      else:
        params[param_name] = param_values['value']
  return params