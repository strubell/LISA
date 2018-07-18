import tensorflow as tf
import data_converters


class Vocab:
  '''
  Handles creating and caching vocabulary files and tf vocabulary lookup ops for a given data file.
  '''

  def __init__(self, data_filename, data_config, save_dir):
    self.data_filename = data_filename
    self.data_config = data_config
    self.save_dir = save_dir

    self.vocab_names = self.make_vocab_files(self.data_filename, self.data_config, self.save_dir)
    self.vocab_lookups = None


  '''
  Creates tf.contrib.lookup ops for all the vocabs defined in self.data_config.
  
  Args: 
    word_embedding_file: File containing word embedding vocab, with words in the first space-separated column
    
  Returns:
    Map from vocab names to tf.contrib.lookup ops.
  '''
  def create_vocab_lookup_ops(self, word_embedding_file):

    # Don't waste GPU memory with these lookup tables; tell tf to put it on CPU
    with tf.device('/cpu:0'):
      vocab_lookup_ops = {}
      for v in self.vocab_names:
        num_oov = 1 if self.data_config[v] else 0
        vocab_lookup_ops[v] = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (self.save_dir, v),
                                                                      num_oov_buckets=num_oov,
                                                                      key_column_index=0)
      if word_embedding_file:
        embeddings_name = word_embedding_file.split("/")[-1]
        vocab_lookup_ops[embeddings_name] = tf.contrib.lookup.index_table_from_file(word_embedding_file,
                                                                                    num_oov_buckets=1,
                                                                                    key_column_index=0,
                                                                                    delimiter=' ')
    tf.logging.log(tf.logging.INFO, "Created %d vocab lookup ops: %s" %
                   (len(vocab_lookup_ops), str([k for k in vocab_lookup_ops.keys()])))
    return vocab_lookup_ops

  '''
  Gets the cached vocab ops for the given datafile, creating them if they already exist.
  This is needed in order to avoid re-creating duplicate lookup ops for each dataset input_fn, 
  since the lookup ops need to be called lazily from the input_fn in order to end up in the same tf.Graph.
  
  Args:
    word_embedding_file: (Optional) file containing word embedding vocab, with words in the first space-separated column
  
  Returns:
    Map from vocab names to tf.contrib.lookup ops.
    
  '''
  def get_lookup_ops(self, word_embedding_file=None):
    if self.vocab_lookups is None:
      self.vocab_lookups = self.create_vocab_lookup_ops(word_embedding_file)
    return self.vocab_lookups


  '''
  Generates vocab files with counts for all the data with the vocab key
  set to True in data_config. Assumes the input file is in CoNLL format.
  
  Args:
    filename: Name of data file to generate vocab files from
    data_config: Data configuration map
  
  Returns:
    List of names of vocabs created
  '''
  def make_vocab_files(self, filename, data_config, save_dir):

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
            datum_idx = data_config[d]['conll_idx']
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