
def lowercase_converter(data_config, split_line, idx):
  return [split_line[idx].lower()]


def parse_roots_self_loop_converter(data_config, split_line, idx):
  head = int(split_line[idx])
  id = int(data_config['id']['idx'])
  return [str(id if head == 0 else head - 1)]


def strip_conll12_domain_converter(data_config, split_line, idx):
  return [split_line[idx].split("/")[0]]


def conll12_binary_predicates_converter(data_config, split_line, idx):
  return [str(split_line[idx] != '-')]


dispatcher = {
  'parse_roots_self_loop': parse_roots_self_loop_converter,
  'strip_conll12_domain': strip_conll12_domain_converter,
  'conll12_binary_predicates': conll12_binary_predicates_converter,
  'lowercase': lowercase_converter,
}


def dispatch(converter_name):
    try:
      return dispatcher[converter_name]
    except KeyError:
      print('Undefined data converter `%s' % converter_name)
      exit(1)