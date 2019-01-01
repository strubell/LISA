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
    s = s if isinstance(s, str) else s.decode('utf-8')
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


def convert_conll(predicted_roles):
  '''

  :param bio_predicted_roles: sequence of predicted role labels
  :return: sequence of conll-formatted predicted role labels
  '''

  def convert_single(s):
    s = s if isinstance(s, str) else s.decode('utf-8')
    return "*" if s == "_" else "(%s*)" % s

  converted = map(convert_single, predicted_roles)
  return converted


def accuracy_np(predictions, targets, mask, accumulator):

  correct = np.sum(np.multiply(predictions == targets, mask))
  total = np.sum(mask)

  accumulator['correct'] += correct
  accumulator['total'] += total

  accuracy = accumulator['correct'] / accumulator['total']
  return accuracy


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
        word = word if isinstance(word, str) else word.decode('utf-8')
        predicate_str = word if predicate else '-'
        roles_str = '\t'.join(tok_role_labels)
        print("%s\t%s" % (predicate_str, roles_str), file=f)
      print(file=f)


# Write targets file w/ format:
# 0	The	_	_	DET	DET	_	_	2	2	det	det	_	_	_	_	_	_
# 1	economy	_	_	NOUN	NOUN	_	_	4	4	nmod:poss	nmod:poss	_	_	A1	_	_	_
# 2	's	_	_	PART	PART	_	_	2	2	case	case	_	_	_	_	_	_
# 3	temperature	_	_	NOUN	NOUN	_	_	7	7	nsubjpass	nsubjpass	Y	temperature.01	A2	A1	_	_
def write_srl_eval_09(filename, words, predicates, sent_lens, role_labels, parse_heads, parse_labels, pos_tags):
  with open(filename, 'w') as f:
    role_labels_start_idx = 0
    num_predicates_per_sent = np.sum(predicates != '_', -1)

    # for each sentence in the batch
    for sent_words, sent_predicates, sent_len, sent_num_predicates, \
        sent_parse_heads, sent_parse_labels, sent_pos_tags in zip(words, predicates, sent_lens, num_predicates_per_sent,
                                                                  parse_heads, parse_labels, pos_tags):
      # grab predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len array
      sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]

      # this is a list of sent_num_predicates lists of srl role labels
      sent_role_labels = list(map(list, zip(*[convert_conll(j[:sent_len]) for j in sent_role_labels_bio])))
      role_labels_start_idx += sent_num_predicates

      # for each token in the sentence
      for j, (word, predicate, parse_head, parse_label, pos_tag) in enumerate(zip(sent_words[:sent_len],
                                                                                  sent_predicates[:sent_len],
                                                                                  sent_parse_heads[:sent_len],
                                                                                  sent_parse_labels[:sent_len],
                                                                                  sent_pos_tags[:sent_len])):
        tok_role_labels = sent_role_labels[j] if sent_role_labels else []
        predicate = predicate if isinstance(predicate, str) else predicate.decode('utf-8')
        word = word if isinstance(word, str) else word.decode('utf-8')
        predicate_str = "Y\t%s:%s" % (word, predicate) if predicate != "_" else '_\t_'
        roles_str = '\t'.join(tok_role_labels)
        print("%s\t%s\t_\t_\t%s\t%s\t_\t_\t%s\t%s\t%s\t%s\t%s\t%s" % (
          j, word, pos_tag, pos_tag, parse_head, parse_head, parse_label, parse_label, predicate_str, roles_str))
      print(file=f)


# Write to this format for eval.pl:
# 1       The             _       DT      _       _       2       det
# 2       economy         _       NN      _       _       4       poss
# 3       's              _       POS     _       _       2       possessive
# 4       temperature     _       NN      _       _       7       nsubjpass
# 5       will            _       MD      _       _       7       aux
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
        token_outputs = (j,
                         word if isinstance(word, str) else word.decode('utf-8'),
                         pos_tag if isinstance(pos_tag, str) else pos_tag.decode('utf-8'),
                         int(parse_head),
                         parse_label if isinstance(parse_label, str) else parse_label.decode('utf-8'))
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


def conll_srl_eval(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file, pos_predictions=None, pos_targets=None):

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
      # print(srl_eval)
      correct, excess, missed = map(int, srl_eval.split('\n')[6].split()[1:4])
    except CalledProcessError as e:
      tf.logging.log(tf.logging.ERROR, "Call to srl-eval.pl (conll srl eval) failed.")

  return correct, excess, missed


def conll09_srl_eval(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                     parse_label_predictions, parse_head_predictions, parse_label_targets, parse_head_targets,
                     pos_targets, pos_predictions, pred_srl_eval_file, gold_srl_eval_file):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  # import time
  # debug_fname = pred_srl_eval_file.decode('utf-8') + str(time.time())
  # write_srl_debug(debug_fname, words, predicate_targets, sent_lens, srl_targets, pos_predictions, pos_targets)

  # write gold labels
  write_srl_eval_09(gold_srl_eval_file, words, predicate_targets, sent_lens, srl_targets, parse_head_targets,
                    parse_label_targets, pos_targets)

  # write predicted labels
  write_srl_eval_09(pred_srl_eval_file, words, predicate_predictions, sent_lens, srl_predictions,
                    parse_head_predictions, parse_label_predictions, pos_predictions)

  # run eval script
  correct, excess, missed = 0, 0, 0
  with open(os.devnull, 'w') as devnull:
    try:
      srl_eval = check_output(["perl", "bin/eval09.pl", "-g", gold_srl_eval_file, "-s", pred_srl_eval_file], stderr=devnull)
      srl_eval = srl_eval.decode('utf-8')
      print(srl_eval)
      correct, excess, missed = map(int, srl_eval.split('\n')[6].split()[1:4])
    except CalledProcessError as e:
      tf.logging.log(tf.logging.ERROR, "Call to eval09.pl (conll09 srl eval) failed.")

  return correct, excess, missed


def conll_parse_eval(parse_label_predictions, parse_head_predictions, words, mask, parse_label_targets,
                        parse_head_targets, pred_eval_file, gold_eval_file, pos_targets):

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
      eval_str = eval.decode('utf-8')

      # Labeled attachment score: 26444 / 29058 * 100 = 91.00 %
      # Unlabeled attachment score: 27251 / 29058 * 100 = 93.78 %
      # Label accuracy score: 27395 / 29058 * 100 = 94.28 %
      first_three_lines = eval_str.split('\n')[:3]
      total = int(first_three_lines[0].split()[5])
      labeled_correct, unlabeled_correct, label_correct = map(lambda l: int(l.split()[3]), first_three_lines)
    except CalledProcessError as e:
      tf.logging.log(tf.logging.ERROR, "Call to eval.pl (conll parse eval) failed.")

  return total, np.array([labeled_correct, unlabeled_correct, label_correct])


def conll_srl_eval_np(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                   gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, accumulator):

  # first, use reverse maps to convert ints to strings
  str_srl_predictions = [list(map(reverse_maps['srl'].get, s)) for s in predictions]
  str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
  str_srl_targets = [list(map(reverse_maps['srl'].get, s)) for s in targets]

  correct, excess, missed = conll_srl_eval(str_srl_predictions, predicate_predictions, str_words, mask, str_srl_targets,
                                           predicate_targets, pred_srl_eval_file, gold_srl_eval_file)

  accumulator['correct'] += correct
  accumulator['excess'] += excess
  accumulator['missed'] += missed

  precision = accumulator['correct'] / (accumulator['correct'] + accumulator['excess'])
  recall = accumulator['correct'] / (accumulator['correct'] + accumulator['missed'])
  f1 = 2 * precision * recall / (precision + recall)

  return f1


def conll09_srl_eval_np(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                        gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, parse_head_predictions,
                        parse_head_targets, parse_label_predictions, parse_label_targets, accumulator):

  # first, use reverse maps to convert ints to strings
  str_srl_predictions = [list(map(reverse_maps['srl'].get, s)) for s in predictions]
  str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
  str_srl_targets = [list(map(reverse_maps['srl'].get, s)) for s in targets]
  str_pos_targets = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_targets]
  str_pos_predictions = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_predictions]
  str_parse_label_targets = [list(map(reverse_maps['parse_label'].get, s)) for s in parse_label_targets]
  str_parse_label_predictions = [list(map(reverse_maps['parse_label'].get, s)) for s in parse_label_predictions]
  str_predicate_predictions = [list(map(reverse_maps['predicate'].get, s)) for s in predicate_predictions]
  str_predicate_targets = [list(map(reverse_maps['predicate'].get, s)) for s in predicate_targets]

  correct, excess, missed = conll09_srl_eval(str_srl_predictions, str_predicate_predictions, str_words, mask,
                                             str_srl_targets, str_predicate_targets, str_parse_label_predictions,
                                             parse_head_predictions, str_parse_label_targets, parse_head_targets,
                                             str_pos_targets, str_pos_predictions, pred_srl_eval_file, gold_srl_eval_file)

  accumulator['correct'] += correct
  accumulator['excess'] += excess
  accumulator['missed'] += missed

  precision = accumulator['correct'] / (accumulator['correct'] + accumulator['excess'])
  recall = accumulator['correct'] / (accumulator['correct'] + accumulator['missed'])
  f1 = 2 * precision * recall / (precision + recall)

  return f1


def conll_parse_eval_np(predictions, targets, parse_head_predictions, words, mask, parse_head_targets, reverse_maps,
                        gold_parse_eval_file, pred_parse_eval_file, pos_targets, accumulator):

  # first, use reverse maps to convert ints to strings
  str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
  str_predictions = [list(map(reverse_maps['parse_label'].get, s)) for s in predictions]
  str_targets = [list(map(reverse_maps['parse_label'].get, s)) for s in targets]
  str_pos_targets = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_targets]

  total, corrects = conll_parse_eval(str_predictions, parse_head_predictions, str_words, mask, str_targets,
                                     parse_head_targets, pred_parse_eval_file, gold_parse_eval_file, str_pos_targets)

  accumulator['total'] += total
  accumulator['corrects'] += corrects

  accuracies = accumulator['corrects'] / accumulator['total']

  return accuracies


fn_dispatcher = {
  'accuracy': accuracy_np,
  'conll_srl_eval': conll_srl_eval_np,
  'conll_parse_eval': conll_parse_eval_np,
  'conll09_srl_eval': conll09_srl_eval_np
}


accumulator_factory = {
  'accuracy': lambda: {'correct': 0., 'total': 0.},
  'conll_srl_eval': lambda: {'correct': 0., 'excess': 0., 'missed': 0.},
  'conll_parse_eval': lambda: {'total': 0., 'corrects': np.zeros(3)},
}


def dispatch(fn_name):
  try:
    return fn_dispatcher[fn_name]
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)


def get_accumulator(fn_name):
  try:
    return accumulator_factory[fn_name]()
  except KeyError:
    print('Undefined evaluation function `%s' % fn_name)
    exit(1)


def get_accumulators(task_config):
  eval_accumulators = {}
  # for i in layer_task_config:
  for task, task_map in task_config.items():
    for eval_name, eval_map in task_map['eval_fns'].items():
      eval_accumulators[eval_name] = get_accumulator(eval_map['name'])
  return eval_accumulators


def get_params(task, task_map, predictions, features, labels, reverse_maps, tokens_to_keep):
  # always pass through predictions, targets and mask
  params = {'predictions': predictions['%s_predictions' % task], 'targets': labels[task], 'mask': tokens_to_keep}
  if 'params' in task_map:
    params_map = task_map['params']
    for param_name, param_values in params_map.items():
      if 'reverse_maps' in param_values:
        params[param_name] = {map_name: reverse_maps[map_name] for map_name in param_values['reverse_maps']}
      elif 'label' in param_values:
        params[param_name] = labels[param_values['label']]
      elif 'feature' in param_values:
        params[param_name] = features[param_values['feature']]
      elif 'layer' in param_values:
        params[param_name] = predictions['%s_%s' % (param_values['layer'], param_values['output'])]
      else:
        params[param_name] = param_values['value']
  return params