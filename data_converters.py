import constants


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


def joint_converter(split_line, idx, component_converters):
  components = [dispatch(converter)(split_line, idx) for i, converter in zip(idx, component_converters)]
  return [constants.JOINT_LABEL_SEP.join(components)]


def idx_range_converter(split_line, idx):
  return split_line[idx[0]: idx[1]]


def idx_list_converter(split_line, idx):
  if isinstance(idx, int):
    return [split_line[idx]]
  return [split_line[i] for i in idx]


dispatcher = {
  'parse_roots_self_loop': parse_roots_self_loop_converter,
  'strip_conll12_domain': strip_conll12_domain_converter,
  'conll12_binary_predicates': conll12_binary_predicates_converter,
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
