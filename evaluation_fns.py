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
    s = s.decode('utf-8')
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


# Write targets file w/ format:
# -        (A1*  (A1*
# -          *     *
# -          *)    *)
# -          *     *
# expected (V*)    *
# -        (C-A1*  *
# widen     *     (V*)
# -         *     (A4*
def write_srl_eval(filename, words, predicates, sent_lens, role_labels):
  with open(filename, 'w') as f:
    role_labels_start_idx = 0
    for sent_words, sent_predicates, sent_len in zip(words, predicates, sent_lens):
      # first get number of predicates
      sent_num_predicates = np.sum(sent_predicates)

      # grab those predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len tensor
      sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]
      sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
      role_labels_start_idx += sent_num_predicates
      predicate_idx = 0
      print("sent num predicates", sent_num_predicates)
      for word, predicate in zip(sent_words[:sent_len], sent_predicates[:sent_len]):
        predicate_str = word.decode('utf-8') if predicate else '-'
        roles_str = ('\t'.join(sent_role_labels[predicate_idx]) if predicate else '')
        predicate_idx += 1 if predicate else 0
        print("%s\t%s" % (predicate_str, roles_str), file=f)
      print(file=f)


def conll_srl_eval_py(predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  print("predictions shape", predictions.shape)
  print("targets shape", srl_targets.shape)

  print("predicate_predictions", predicate_predictions)
  print("predicate_targets", predicate_targets)

  print("predictions", predictions)
  print("targets", srl_targets)


  overall_f1 = 0.0


  # write gold labels
  write_srl_eval(gold_srl_eval_file, words, predicate_targets, sent_lens, srl_targets)

  # write predicted labels
  write_srl_eval(pred_srl_eval_file, words, predicate_predictions, sent_lens, predictions)

  # run eval script
  with open(os.devnull, 'w') as devnull:
    try:
      srl_eval = check_output(["perl", "bin/srl-eval.pl", gold_srl_eval_file, pred_srl_eval_file], stderr=devnull)
      print(srl_eval)
      # todo actually, get all the cumulative counts
      overall_f1 = float(srl_eval.split('\n')[6].split()[-1])
    except CalledProcessError as e:
      print("Call to srl-eval.pl eval failed.")

  return overall_f1, overall_f1, overall_f1


def create_metric_variable(name, shape, dtype):
  return tf.get_variable(name=name, shape=shape, dtype=dtype, collections=[tf.GraphKeys.LOCAL_VARIABLES,
                                                                           tf.GraphKeys.METRIC_VARIABLES])


def conll_srl_eval(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                   gold_srl_eval_file, pred_srl_eval_file):

  # create accumulator variables
  # correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int32)
  # excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int32)
  # missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int32)
  correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.float64)
  excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.float64)
  missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.float64)

  predictions = tf.Print(predictions, [predictions], "predictions")

  # first, use reverse maps to convert ints to strings
  # todo order of map.values() is probably not guaranteed; should prob sort by keys first
  str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), predictions, name="str_srl_preds_lookup")
  str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words, name="str_words_lookup")
  str_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), targets, name="str_srl_targets_lookup")

  str_predictions = tf.Print(str_predictions, [str_predictions], "str predictions")

  # need to pass through the stuff for pyfunc
  py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets, pred_srl_eval_file, gold_srl_eval_file]
  out_types = [tf.float64, tf.float64, tf.float64] # [tf.int32, tf.int32, tf.int32]
  correct, excess, missed = tf.py_func(conll_srl_eval_py, py_eval_inputs, out_types, stateful=False)
  # correct = counts[0]
  # excess = counts[1]
  # missed = counts[2]

  update_correct_op = tf.assign_add(correct_count, correct)
  update_excess_op = tf.assign_add(excess_count, excess)
  update_missed_op = tf.assign_add(missed_count, missed)

  # precision = correct / (excess + correct)
  # recall = correct / (missed + correct)
  # f1 = 2 * precision * recall / (precision + recall)

  precision_update_op = update_correct_op / (update_correct_op + update_excess_op)
  recall_update_op = update_correct_op / (update_correct_op + update_missed_op)
  f1_update_op = 2 * precision_update_op * recall_update_op / (precision_update_op + recall_update_op)

  precision = correct_count / (correct_count + excess_count)
  recall = correct_count / (correct_count + missed_count)
  f1 = 2 * precision * recall / (precision + recall)

  return f1, f1_update_op


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