import tensorflow as tf
import numpy as np
import os
from subprocess import check_output, CalledProcessError


# todo simplify to convert_bio
def convert_bilou(bio_predicted_roles):
  '''

  :param bio_predicted_roles: sequence of BIO-encoded predicted role labels
  :return: sequence of conll-formatted predicted role labels
  '''

  converted = []
  started_types = []
  for i, s in enumerate(bio_predicted_roles):
    label_parts = s.split('/')
    curr_len = len(label_parts)
    combined_str = ''
    Itypes = []
    Btypes = []
    for idx, label in enumerate(label_parts):
      bilou = label[0]
      label_type = label[2:]
      props_str = ''
      if bilou == 'I':
        Itypes.append(label_type)
        props_str = ''
      elif bilou == 'O':
        curr_len = 0
        props_str = ''
      elif bilou == 'U':
        # need to check whether last one was ended
        props_str = '(' + label_type + ('*)' if idx == len(label_parts) - 1 else "")
      elif bilou == 'B':
        # need to check whether last one was ended
        props_str = '(' + label_type
        started_types.append(label_type)
        Btypes.append(label_type)
      elif bilou == 'L':
        props_str = ')'
        started_types.pop()
        curr_len -= 1
      combined_str += props_str
    while len(started_types) > curr_len:
      converted[-1] += ')'
      started_types.pop()
    while len(started_types) < len(Itypes) + len(Btypes):
      combined_str = '(' + Itypes[-1] + combined_str
      started_types.append(Itypes[-1])
      Itypes.pop()
    if not combined_str:
      combined_str = '*'
    elif combined_str[0] == "(" and combined_str[-1] != ")":
      combined_str += '*'
    elif combined_str[-1] == ")" and combined_str[0] != "(":
      combined_str = '*' + combined_str
    converted.append(combined_str)
  while len(started_types) > 0:
    converted[-1] += ')'
    started_types.pop()
  return converted


def accuracy(predictions, targets, mask):
  return tf.metrics.accuracy(targets, predictions, weights=mask)


def conll_parse_eval(predictions, targets, mask, reverse_maps, gold_parse_eval_file, pred_parse_eval_file):
  # TODO write this function
  return tf.metrics.accuracy(targets, predictions, mask)


def conll_srl_eval_py(predictions, predicate_predictions, words, mask, pred_srl_eval_file, gold_srl_eval_file):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence

  # srl_preds_str = map(list, zip(*[self.convert_bilou(j) for j in np.transpose(srl_preds)]))

  sent_lens = np.sum(mask, -1)

  # First write predictions file w/ format:
  # -        (A1*  (A1*
  # -          *     *
  # -          *)    *)
  # -          *     *
  # expected (V*)    *
  # -        (C-A1*  *
  # widen     *     (V*)
  # -         *     (A4*
  with open(pred_srl_eval_file, 'w') as f:
    predicate_start_idx = 0
    for sent_words, sent_predicates, sent_len in zip(words, predicate_predictions, sent_lens):
      # first get number of predicates
      sent_num_predicates = np.sum(sent_predicates)

      # grab those predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len tensor
      sent_role_preds_bio = predictions[predicate_start_idx: predicate_start_idx+sent_num_predicates]
      sent_role_preds = map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_preds_bio]))
      for word, predicate, role_labels in zip(sent_words, sent_predicates, sent_role_preds):
        predicate_str = word if predicate else '-'
        print("%s\t%s" % (predicate_str, '\t'.join(role_labels)), file=f)

  overall_f1 = 0
  with open(os.devnull, 'w') as devnull:
    try:
      srl_eval = check_output(["perl", "bin/srl-eval.pl", gold_srl_eval_file, pred_srl_eval_file], stderr=devnull)
      print(srl_eval)
      # todo actually, get all the cumulative counts
      overall_f1 = float(srl_eval.split('\n')[6].split()[-1])
    except CalledProcessError as e:
      print("Call to srl-eval.pl eval failed.")

  return overall_f1


def create_metric_variable(name, shape, dtype):
  return tf.get_variable(name=name, shape=shape, dtype=dtype, collections=[tf.GraphKeys.LOCAL_VARIABLES,
                                                                           tf.GraphKeys.METRIC_VARIABLES])

def conll_srl_eval(predictions, targets, predicate_predictions, words, mask, reverse_maps, gold_srl_eval_file,
                   pred_srl_eval_file):

  # create accumulator variables
  correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int32)
  excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int32)
  missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int32)

  # first, use reverse maps to convert ints to strings
  # todo order of map.values() is probably not guaranteed; should prob sort by keys first
  str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), predictions, name="str_srl_preds_lookup")
  str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words, name="str_words_lookup")

  # need to pass through the stuff for pyfunc
  py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, pred_srl_eval_file, gold_srl_eval_file]
  out_types = [tf.int32] # [tf.int32, tf.int32, tf.int32]
  counts = tf.py_func(conll_srl_eval_py, py_eval_inputs, out_types, stateful=False)

  correct_count = tf.assign_add(correct_count, counts)

  # need to call pyfunc to get cumulative scores

  # need to also define how to compute f1 from those

  # return tf.metrics.accuracy(targets, predictions, mask)
  return correct_count, correct_count


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


def get_params(task_outputs, task_map, train_outputs, features, labels, task_labels, lookup_maps, tokens_to_keep):
  targets = task_labels if 'targets' not in task_outputs else task_outputs['targets']
  mask = tokens_to_keep if 'mask' not in task_outputs else task_outputs['mask']
  params = {'predictions': task_outputs['predictions'], 'targets': targets, 'mask': mask}
  if 'params' in task_map:
    params_map = task_map['params']
    for param_name, param_values in params_map.items():
      if 'maps' in param_values:
        params[param_name] = {map_name: lookup_maps[map_name] for map_name in param_values['maps']}
      elif 'label' in param_values:
        params[param_name] = labels[param_values['label']]
      elif 'feature' in param_values:
        params[param_name] = features[param_values['feature']]
      elif 'layer' in param_values:
        outputs_layer = train_outputs[param_values['layer']]
        params[param_name] = outputs_layer[param_values['output']]
      else:
        params[param_name] = param_values['value']
  return params