import tensorflow as tf
import data_converters

'''
Generates vocab files with counts for all the data with the vocab key
set to True in data_config. Assumes the input file is in CoNLL format.

Args:
  filename: Name of data file to generate vocab files from
  data_config: Data configuration map

Returns:
  List of names of vocabs created
'''
def make_vocab_files(filename, data_config, save_dir):

  # init maps
  vocabs = []
  vocabs_index = {}
  for d in data_config:
    if 'vocab' in data_config[d] and data_config[d]['vocab'] == d:
      vocabs.append({})
      vocabs_index[d] = len(vocabs_index)

  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        split_line = line.split()
        for d in vocabs_index.keys():
          datum_idx = data_config[d]['idx']
          this_vocab_map = vocabs[vocabs_index[d]]
          if isinstance(datum_idx, int):
            this_data = [split_line[datum_idx]]
          else:
            this_data = split_line[datum_idx[0]: datum_idx[1]]
          if 'converter' in data_config[d]:
            this_data = data_converters.dispatch(data_config[d]['converter'])(data_config, split_line, datum_idx)
          for this_datum in this_data:
            if this_datum not in this_vocab_map:
              this_vocab_map[this_datum] = 0
            this_vocab_map[this_datum] += 1

  for d in vocabs_index.keys():
    this_vocab_map = vocabs[vocabs_index[d]]
    with open("%s/%s.txt" % (save_dir, d), 'w') as f:
      for k, v in this_vocab_map.items():
        print("%s\t%d" % (k, v), file=f)

  return vocabs_index.keys()


'''

'''
def create_vocab_lookup_ops(data_filename, data_config, args):
  vocab_names = make_vocab_files(data_filename, data_config, args.save_dir)

  with tf.device('/cpu:0'):
    vocab_lookup_ops = {}
    for v in vocab_names:
      num_oov = 1 if data_config[v] else 0
      vocab_lookup_ops[v] = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (args.save_dir, v),
                                                                    num_oov_buckets=num_oov,
                                                                    key_column_index=0)
    if args.word_embedding_file:
      embeddings_name = args.word_embedding_file.split("/")[-1]
      vocab_lookup_ops[embeddings_name] = tf.contrib.lookup.index_table_from_file(args.word_embedding_file,
                                                                                  num_oov_buckets=1,
                                                                                  key_column_index=0,
                                                                                  delimiter=' ')
  tf.logging.log(tf.logging.INFO, "Created %d vocab lookup ops: %s" %
                 (len(vocab_lookup_ops), str([k for k in vocab_lookup_ops.keys()])))
  return vocab_lookup_ops