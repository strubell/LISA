import data_converters
import example_converters


def conll_data_generator(filenames, data_config):
  """Converts a stream of lines from a one-word-per-line data file to a stream of lists of processed
     fields from each line, with processing as defined in data_config / data_converters.py.
  """

  example_converter_name = data_config["example_converter"] if "example_converter" in data_config else "default_converter"
  mapping_config = data_config["mappings"]

  for filename in filenames:
    with open(filename, 'r') as f:
      sents = 0
      toks = 0
      buf = []
      for line in f:
        line = line.strip()
        if line:
          toks += 1
          split_line = line.split()
          data_vals = []
          for d, config in mapping_config.items():
            # only return the data that we're actually going to use as inputs or outputs
            if ('feature' in config and config['feature']) or ('label' in config and config['label']):
              datum_idx = config['conll_idx']
              converter_name = config['converter']['name'] if 'converter' in config else 'default_converter'
              converter_params = data_converters.get_params(config, split_line, datum_idx)
              data = data_converters.dispatch(converter_name)(**converter_params)
              data_vals.extend(data)
          buf.append(tuple(data_vals))
        else:
          if buf:
            sents += 1
            converted_example = example_converters.dispatch(example_converter_name)(buf)
            yield converted_example
            buf = []
          # print()
      # catch the last one
      if buf:
        sents += 1
        converted_example = example_converters.dispatch(example_converter_name)(buf)
        yield converted_example



# def bert_data_generator(filenames, data_config):
#   """Converts a stream of lines from a one-word-per-line data file to a stream of lists of processed
#      fields from each line, with processing as defined in data_config / data_converters.py.
#   """
#   for filename in filenames:
#     with open(filename, 'r') as f:
#       sents = 0
#       toks = 0
#       buf = []
#       for line in f:
#         line = line.strip()
#         if line:
#           toks += 1
#           split_line = line.split()
#           data_vals = []
#           for d in data_config.keys():
#             # only return the data that we're actually going to use as inputs or outputs
#             if ('feature' in data_config[d] and data_config[d]['feature']) or \
#                ('label' in data_config[d] and data_config[d]['label']):
#               datum_idx = data_config[d]['conll_idx']
#               converter_name = data_config[d]['converter']['name'] if 'converter' in data_config[d] else 'default_converter'
#               converter_params = data_converters.get_params(data_config[d], split_line, datum_idx)
#               data = data_converters.dispatch(converter_name)(**converter_params)
#               data_vals.extend(data)
#           # print(tuple(data_vals))
#           # buf.append(tuple(data_vals))
#           buf.extend(data_vals)
#         else:
#           if buf:
#             # todo do this automatically
#             buf = ['[CLS]'] + buf + ['[SEP]']
#             sents += 1
#             yield buf
#             buf = []
#           # print()
#       # catch the last one
#       if buf:
#         yield buf
