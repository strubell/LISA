

def bert_sent_converter(example_buf):
  ''' Flatten and add [CLS] and [SEP] tokens. '''
  flat_buf = [item for sublist in example_buf for item in sublist]
  example_buf = ['[CLS]'] + flat_buf + ['[SEP]']

  # transpose
  return [[e] for e in example_buf]
  # return [example_buf]


def noop_converter(example_buf):
  return example_buf


dispatcher = {
  'bert_sent_converter': bert_sent_converter,
  'default_converter': noop_converter
}


def dispatch(converter_name):
    try:
      return dispatcher[converter_name]
    except KeyError:
      print('Undefined example converter: %s' % converter_name)
      exit(1)
