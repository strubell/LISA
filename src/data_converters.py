import constants
import bert.tokenization

tokenizers = {}


def get_wordpiece_tokenizer(vocab):
  if vocab not in tokenizers:
    tokenizer = bert.tokenization.WordpieceTokenizer(bert.tokenization.load_vocab(vocab))
    tokenizers[vocab] = tokenizer
  return tokenizers[vocab]


def lowercase_converter(split_line, idx):
  return [split_line[idx].lower()]


def parse_roots_self_loop_converter(split_line, idx):
  # idx[0] is parse head, idx[1] is token id
  head = int(split_line[idx[0]])
  id = int(split_line[idx[1]])
  return [str(id if head == 0 else head - 1)]


def strip_conll12_domain_converter(split_line, idx):
  return [split_line[idx].split("/")[0]]


def conll12_binary_predicates_converter(split_line, idx):
  return [str(split_line[idx] != '-')]


def conll09_binary_predicates_converter(split_line, idx):
  return [str(split_line[idx] != '_')]


def conll09_predicate_sense_converter(split_line, idx):
  verb_sense_str = split_line[idx]
  just_verb_sense = verb_sense_str if verb_sense_str == "_" else verb_sense_str.split('.')[1]
  return [just_verb_sense]


def joint_converter(split_line, idx, component_converters):
  components = [dispatch(converter)(split_line, i)[0] for i, converter in zip(idx, component_converters)]
  return [constants.JOINT_LABEL_SEP.join(components)]


def idx_range_converter(split_line, idx):
  return split_line[idx[0]: (idx[1] if idx[1] != -1 else len(split_line))]


def idx_list_converter(split_line, idx):
  if isinstance(idx, int):
    return [split_line[idx]]
  return [split_line[i] for i in idx]


def bert_wordpiece_converter(split_line, idx, wordpiece_vocab):
  word = split_line[idx]
  tokenizer = get_wordpiece_tokenizer(wordpiece_vocab)
  wordpieces = tokenizer.tokenize(word)
  # num_pieces = str(len(wordpieces))
  return wordpieces


def bert_wordpiece_lens_converter(split_line, idx, wordpiece_vocab):
  word = split_line[idx]
  tokenizer = get_wordpiece_tokenizer(wordpiece_vocab)
  wordpieces = tokenizer.tokenize(word)
  num_pieces = str(len(wordpieces))
  return [num_pieces]


dispatcher = {
  'bert_wordpiece_converter': bert_wordpiece_converter,
  'bert_wordpiece_lens_converter': bert_wordpiece_lens_converter,
  'parse_roots_self_loop': parse_roots_self_loop_converter,
  'strip_conll12_domain': strip_conll12_domain_converter,
  'conll12_binary_predicates': conll12_binary_predicates_converter,
  'conll09_binary_predicates': conll09_binary_predicates_converter,
  'conll09_predicate_sense': conll09_predicate_sense_converter,
  'lowercase': lowercase_converter,
  'joint_converter': joint_converter,
  'idx_range_converter': idx_range_converter,
  'idx_list_converter': idx_list_converter,
  'default_converter': idx_list_converter
}


def get_params(datum_config, split_line, idx):
  params = {'split_line': split_line, 'idx': idx}
  if 'converter' in datum_config and 'params' in datum_config['converter']:
    params_map = datum_config['converter']['params']
    for param_name, param_value in params_map.items():
      params[param_name] = param_value
  return params


def dispatch(converter_name):
    try:
      return dispatcher[converter_name]
    except KeyError:
      print('Undefined data converter: %s' % converter_name)
      exit(1)
