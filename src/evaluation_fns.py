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
  with tf.name_scope('accuracy'):
    return tf.metrics.accuracy(targets, predictions, weights=mask)


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
    num_predicates_per_sent = np.sum(predicates, -1)
    # for each sentence in the batch
    for sent_words, sent_predicates, sent_len, sent_num_predicates in zip(words, predicates, sent_lens,
                                                                          num_predicates_per_sent):
      # grab predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len array
      sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]

      # this is a list of sent_num_predicates lists of srl role labels
      sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
      role_labels_start_idx += sent_num_predicates

      # for each token in the sentence
      for j, (word, predicate) in enumerate(zip(sent_words[:sent_len], sent_predicates[:sent_len])):
        tok_role_labels = sent_role_labels[j] if sent_role_labels else []
        predicate_str = word.decode('utf-8') if predicate else '-'
        roles_str = '\t'.join(tok_role_labels)
        print("%s\t%s" % (predicate_str, roles_str), file=f)
      print(file=f)


# Write to this format for eval.pl:
# 1       The     _       DT      _       _       2       det
# 2       economy _       NN      _       _       4       poss
# 3       's      _       POS     _       _       2       possessive
# 4       temperature     _       NN      _       _       7       nsubjpass
# 5       will    _       MD      _       _       7       aux
def write_parse_eval(filename, words, parse_heads, sent_lens, parse_labels, pos_tags):
  with open(filename, 'w') as f:

    # for each sentence in the batch
    for sent_words, sent_parse_heads, sent_len, sent_parse_labels, sent_pos_tags in zip(words, parse_heads, sent_lens,
                                                                                        parse_labels, pos_tags):
      # for each token in the sentence
      for j, (word, parse_head, parse_label, pos_tag) in enumerate(zip(sent_words[:sent_len],
                                                                       sent_parse_heads[:sent_len],
                                                                       sent_parse_labels[:sent_len],
                                                                       sent_pos_tags[:sent_len])):
        parse_head = 0 if j == parse_head else parse_head + 1
        token_outputs = (j, word.decode('utf-8'), pos_tag.decode('utf-8'), int(parse_head), parse_label.decode('utf-8'))
        print("%d\t%s\t_\t%s\t_\t_\t%d\t%s" % token_outputs, file=f)
      print(file=f)


def write_srl_debug(filename, words, predicates, sent_lens, role_labels, pos_predictions, pos_targets):
  with open(filename, 'w') as f:
    role_labels_start_idx = 0
    num_predicates_per_sent = np.sum(predicates, -1)
    # for each sentence in the batch
    for sent_words, sent_predicates, sent_len, sent_num_predicates, pos_preds, pos_targs in zip(words, predicates, sent_lens,
                                                                          num_predicates_per_sent, pos_predictions,
                                                                          pos_targets):
      # grab predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len array
      sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]

      # this is a list of sent_num_predicates lists of srl role labels
      sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
      role_labels_start_idx += sent_num_predicates

      sent_role_labels_bio = list(zip(*sent_role_labels_bio))

      pos_preds = list(map(lambda d: d.decode('utf-8'), pos_preds))
      pos_targs = list(map(lambda d: d.decode('utf-8'), pos_targs))

      # for each token in the sentence
      # printed = False
      for j, (word, predicate, pos_p, pos_t) in enumerate(zip(sent_words[:sent_len], sent_predicates[:sent_len],
                                                              pos_preds[:sent_len], pos_targs[:sent_len])):
        tok_role_labels = sent_role_labels[j] if sent_role_labels else []
        bio_tok_role_labels = sent_role_labels_bio[j][:sent_len] if sent_role_labels else []
        word_str = word.decode('utf-8')
        predicate_str = str(predicate)
        roles_str = '\t'.join(tok_role_labels)
        bio_roles_str = '\t'.join(map(lambda d: d.decode('utf-8'), bio_tok_role_labels))
        print("%s\t%s\t%s\t%s\t%s\t%s" % (word_str, predicate_str, pos_t, pos_p, roles_str, bio_roles_str), file=f)
      print(file=f)


def conll_srl_eval_py(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file, pos_predictions, pos_targets):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  # import time
  # debug_fname = pred_srl_eval_file.decode('utf-8') + str(time.time())
  # write_srl_debug(debug_fname, words, predicate_targets, sent_lens, srl_targets, pos_predictions, pos_targets)

  # write gold labels
  write_srl_eval(gold_srl_eval_file, words, predicate_targets, sent_lens, srl_targets)

  # write predicted labels
  write_srl_eval(pred_srl_eval_file, words, predicate_predictions, sent_lens, srl_predictions)

  # run eval script
  correct, excess, missed = 0, 0, 0
  with open(os.devnull, 'w') as devnull:
    try:
      srl_eval = check_output(["perl", "bin/srl-eval.pl", gold_srl_eval_file, pred_srl_eval_file], stderr=devnull)
      srl_eval = srl_eval.decode('utf-8')
      print(srl_eval)
      correct, excess, missed = map(int, srl_eval.split('\n')[6].split()[1:4])
    except CalledProcessError as e:
      print("Call to srl-eval.pl eval failed.")

  return correct, excess, missed


def create_metric_variable(name, shape, dtype):
  return tf.get_variable(name=name, shape=shape, dtype=dtype, collections=[tf.GraphKeys.LOCAL_VARIABLES,
                                                                           tf.GraphKeys.METRIC_VARIABLES])


def conll_srl_eval(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                   gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets):

  with tf.name_scope('conll_srl_eval'):

    # create accumulator variables
    correct_count = create_metric_variable("correct_count", shape=[], dtype=tf.int64)
    excess_count = create_metric_variable("excess_count", shape=[], dtype=tf.int64)
    missed_count = create_metric_variable("missed_count", shape=[], dtype=tf.int64)

    # first, use reverse maps to convert ints to strings
    # todo order of map.values() is probably not guaranteed; should prob sort by keys first
    str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), predictions)
    str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words)
    str_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['srl'].values())), targets)

    str_pos_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_predictions)
    str_pos_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_targets)

    # need to pass through the stuff for pyfunc
    # pyfunc is necessary here since we need to write to disk
    py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file, str_pos_predictions, str_pos_targets]
    out_types = [tf.int64, tf.int64, tf.int64]
    correct, excess, missed = tf.py_func(conll_srl_eval_py, py_eval_inputs, out_types, stateful=False)

    update_correct_op = tf.assign_add(correct_count, correct)
    update_excess_op = tf.assign_add(excess_count, excess)
    update_missed_op = tf.assign_add(missed_count, missed)

    precision_update_op = update_correct_op / (update_correct_op + update_excess_op)
    recall_update_op = update_correct_op / (update_correct_op + update_missed_op)
    f1_update_op = 2 * precision_update_op * recall_update_op / (precision_update_op + recall_update_op)

    precision = correct_count / (correct_count + excess_count)
    recall = correct_count / (correct_count + missed_count)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, f1_update_op


def conll_parse_eval_py(parse_label_predictions, parse_head_predictions, words, mask, parse_label_targets,
                        parse_head_targets, pred_eval_file, gold_eval_file, pos_predictions, pos_targets):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  # write gold labels
  write_parse_eval(gold_eval_file, words, parse_head_targets, sent_lens, parse_label_targets, pos_targets)

  # write predicted labels
  write_parse_eval(pred_eval_file, words, parse_head_predictions, sent_lens, parse_label_predictions, pos_targets)

  # run eval script
  total, labeled_correct, unlabeled_correct, label_correct = 0, 0, 0, 0
  with open(os.devnull, 'w') as devnull:
    try:
      eval = check_output(["perl", "bin/eval.pl", "-g", gold_eval_file, "-s", pred_eval_file], stderr=devnull)
      eval = eval.decode('utf-8')

      # Labeled attachment score: 26444 / 29058 * 100 = 91.00 %
      # Unlabeled attachment score: 27251 / 29058 * 100 = 93.78 %
      # Label accuracy score: 27395 / 29058 * 100 = 94.28 %
      first_three_lines = eval.split('\n')[:3]
      total = int(first_three_lines[0].split()[5])
      labeled_correct, unlabeled_correct, label_correct = map(lambda l: int(l.split()[3]), first_three_lines)
      # correct, excess, missed = map(int, srl_eval.split('\n')[6].split()[1:4])
    except CalledProcessError as e:
      print("Call to eval.pl eval failed.")

  return total, np.array([labeled_correct, unlabeled_correct, label_correct])


# todo share computation with srl eval
def conll_parse_eval(predictions, targets, parse_head_predictions, words, mask, parse_head_targets, reverse_maps,
                   gold_parse_eval_file, pred_parse_eval_file, pos_predictions, pos_targets):

  with tf.name_scope('conll_parse_eval'):

    # create accumulator variables
    total_count = create_metric_variable("total_count", shape=[], dtype=tf.int64)
    # labeled_correct = create_metric_variable("labeled_correct", shape=[], dtype=tf.int64)
    # unlabeled_correct = create_metric_variable("unlabeled_correct", shape=[], dtype=tf.int64)
    # label_correct = create_metric_variable("label_correct", shape=[], dtype=tf.int64)
    correct_count = create_metric_variable("label_correct", shape=[3], dtype=tf.int64)

    # first, use reverse maps to convert ints to strings
    # todo order of map.values() is probably not guaranteed; should prob sort by keys first
    str_words = tf.nn.embedding_lookup(np.array(list(reverse_maps['word'].values())), words)
    str_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())), predictions)
    str_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['parse_label'].values())), targets)

    str_pos_predictions = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_predictions)
    str_pos_targets = tf.nn.embedding_lookup(np.array(list(reverse_maps['gold_pos'].values())), pos_targets)

    # need to pass through the stuff for pyfunc
    # pyfunc is necessary here since we need to write to disk
    py_eval_inputs = [str_predictions, parse_head_predictions, str_words, mask, str_targets, parse_head_targets,
                      pred_parse_eval_file, gold_parse_eval_file, str_pos_predictions, str_pos_targets]
    out_types = [tf.int64, tf.int64] #, tf.int64, tf.int64]
    total, corrects = tf.py_func(conll_parse_eval_py, py_eval_inputs, out_types, stateful=False)

    update_total_count_op = tf.assign_add(total_count, total)
    update_correct_op = tf.assign_add(correct_count, corrects)

    update_op = update_correct_op / update_total_count_op

    accuracies = corrects / total

    # update_total_count_op = tf.assign_add(total_count, total)
    # update_labeled_correct_op = tf.assign_add(labeled_correct, labeled)
    # update_unlabeled_correct_op = tf.assign_add(unlabeled_correct, unlabeled)
    # update_label_correct_op = tf.assign_add(label_correct, label)

    # uas_update_op = update_labeled_correct_op / update_total_count_op
    # las_update_op = update_unlabeled_correct_op / update_total_count_op
    # ls_update_op = update_label_correct_op / update_total_count_op

    # uas = labeled_correct / total_count
    # las = unlabeled_correct / total_count
    # ls = label_correct / total_count

    return accuracies, update_op


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

  # always pass through predictions, targets and mask
  params = {'predictions': task_outputs['predictions'], 'targets': task_labels, 'mask': tokens_to_keep}
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