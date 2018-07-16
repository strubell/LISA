import tensorflow as tf
import data_converters

def conll_data_generator(filename, data_config):
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
        for d in data_config.keys():
          # only return the data that we're actually going to use as inputs or outputs
          if ('feature' in data_config[d] and data_config[d]['feature']) or \
             ('label' in data_config[d] and data_config[d]['label']):
            idx = data_config[d]['idx']
            data = [split_line[idx]] if isinstance(idx, int) else split_line[idx[0]:idx[1]]
            if 'converter' in data_config[d]:
              data = data_converters.dispatch(data_config[d]['converter'])(data_config, split_line, idx)
            data_vals.extend(data)
        # print(data_vals)
        buf.append(tuple(data_vals))
      else:
        if buf:
          sents += 1
          yield buf
          buf = []
    # catch the last one
    if buf:
      yield buf
