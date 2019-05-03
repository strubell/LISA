import tensorflow as tf
import numpy as np
import collections
import os
import util
import constants
import data_converters


class Vocab:
  '''
  Handles creating and caching vocabulary files and tf vocabulary lookup ops for a given list of data files.
  '''

  def __init__(self, data_configs, save_dir, data_filenames=None, embedding_files=None):
    self.data_configs = data_configs
    self.save_dir = save_dir

    # self.vocab_sizes = {}
    self.joint_label_lookup_maps = {}
    self.reverse_maps = {}
    self.vocab_maps = {}
    self.vocab_lookups = None
    self.oovs = {}

    # make directory for vocabs
    self.vocabs_dir = "%s/assets.extra" % save_dir
    if not os.path.exists(self.vocabs_dir):
      try:
        os.mkdir(self.vocabs_dir)
      except OSError as e:
        util.fatal_error("Failed to create vocabs directory: %s; %s" % (self.vocabs_dir, e.strerror))
      else:
        tf.logging.info("Successfully created vocabs directory: %s" % self.vocabs_dir)
    else:
      tf.logging.info("Using vocabs directory: %s" % self.vocabs_dir)

    self.vocab_names_sizes, self.combined_data_config = self.make_vocab_files(self.data_configs, data_filenames, embedding_files)



  '''
  Creates tf.contrib.lookup ops for all the vocabs defined in self.data_config.
    
  Returns:
    Map from vocab names to tf.contrib.lookup ops, map from vocab names to vocab sizes
  '''
  def create_vocab_lookup_ops(self): # , embedding_files=None):

    # Don't waste GPU memory with these lookup tables; tell tf to put it on CPU
    with tf.device('/cpu:0'):
      vocab_lookup_ops = {}
      for v in self.vocab_names_sizes.keys():
        # if v in self.data_config:
        v_clean = util.clean_filename(v)
        num_oov = 1 if 'oov' in self.combined_data_config[v] and self.combined_data_config[v]['oov'] else 0
        this_lookup = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (self.vocabs_dir, v_clean),
                                                              num_oov_buckets=num_oov,
                                                              key_column_index=0)
        vocab_lookup_ops[v] = this_lookup

        # add OOV as last element to in-memory dicts if we need to. this happens here rather than when creating
        # the maps so that we don't load redundant OOVs into lookup table above, and when updating vocabs.
        # if num_oov:
        #   this_vocab_map = self.vocab_maps[v]
        #   this_reverse_map = self.reverse_maps[v]
        #   oov_idx = len(this_vocab_map)
        #   this_vocab_map[constants.OOV_STRING] = oov_idx
        #   this_reverse_map[oov_idx] = constants.OOV_STRING
        #   self.vocab_names_sizes[v] += 1

      # if embedding_files:
      #   for embedding_file in embedding_files:
      #     embeddings_name = embedding_file
      #     vocab_lookup_ops[embeddings_name] = tf.contrib.lookup.index_table_from_file(embedding_file,
      #                                                                                 num_oov_buckets=1,
      #                                                                                 key_column_index=0,
      #                                                                                 delimiter=' ')
      #     # annoying, but lookup.size() returns a tensor, so read the length of the file
      #     # self.vocab_names_sizes[embeddings_name] = vocab_lookup_ops[embeddings_name].size()
      #     self.vocab_names_sizes[embeddings_name] = util.lines_in_file(embedding_file)

    tf.logging.info("Created %d vocab lookup ops: %s" %
                    (len(vocab_lookup_ops), str([k for k in vocab_lookup_ops.keys()])))
    return vocab_lookup_ops

  # '''
  # Gets the cached vocab ops for the given datafile, creating them if they already exist.
  # This is needed in order to avoid re-creating duplicate lookup ops for each dataset input_fn,
  # since the lookup ops need to be called lazily from the input_fn in order to end up in the same tf.Graph.
  #
  # Returns:
  #   Map from vocab names to tf.contrib.lookup ops.
  #
  # '''
  # def get_lookup_ops(self):
  #   if self.vocab_lookups is None:
  #     self.vocab_lookups = self.create_vocab_lookup_ops()
  #   return self.vocab_lookups


  '''
  Generates vocab files with counts for all the data with the vocab key
  specified in data_config. Assumes the input file is in CoNLL format.
  
  Args:
    filename: Name of data file to generate vocab files from
    data_config: Data configuration map
  
  Returns:
    Map from vocab names to their sizes
  '''
  def create_load_or_update_vocab_files(self, data_configs, filenames=None, embedding_files=None, update_only=False):

    # init maps
    vocabs = []
    vocabs_index = {}
    combined_data_config = {}
    vocabs_from_file = set()
    idx = 0
    for data_config in data_configs:
      for c in data_config:
        mapping_config = data_config[c]['mappings']
        for d, config_map in mapping_config.items():
          updatable = 'updatable' in config_map and config_map['updatable']
          # if 'vocab' in data_config[d] and data_config[d]['vocab'] == d and (updatable or not update_only):
          if 'vocab' in config_map and (updatable or not update_only):
            this_vocab_name = config_map['vocab']
            # if the vocab name is the same as the name of this entry, then we create our own vocab from the data
            if this_vocab_name == d:
              vocabs_from_file.add(this_vocab_name)
            combined_data_config[this_vocab_name] = config_map
            this_vocab = collections.OrderedDict()
            if update_only and updatable and this_vocab_name in self.vocab_maps:
              this_vocab = self.vocab_maps[this_vocab_name]
            vocabs.append(this_vocab)
            vocabs_index[this_vocab_name] = idx
            idx += 1

    # Create vocabs from data files
    if filenames:
      for filename in filenames:
        with open(filename, 'r') as f:
          for line in f:
            line = line.strip()
            if line:
              split_line = line.split()
              # only want to do this for vocabs that we're generating
              for d in vocabs_from_file:
                datum_idx = combined_data_config[d]['conll_idx']
                this_vocab_map = vocabs[vocabs_index[d]]
                converter_name = combined_data_config[d]['converter']['name'] if 'converter' in combined_data_config[d] else 'default_converter'
                converter_params = data_converters.get_params(combined_data_config[d], split_line, datum_idx)
                this_data = data_converters.dispatch(converter_name)(**converter_params)
                for this_datum in this_data:
                  if this_datum not in this_vocab_map:
                    this_vocab_map[this_datum] = 0
                  this_vocab_map[this_datum] += 1

    # Assume we have the vocabs saved to disk; load them
    else:
      for d in vocabs_index.keys():
        # if d in combined_data_config: # todo i dont think this line is necessary
        this_vocab_map = vocabs[vocabs_index[d]]
        with open("%s/%s.txt" % (self.vocabs_dir, d), 'r') as f:
          for line in f:
            datum, count = line.strip().split()
            this_vocab_map[datum] = int(count)

    if embedding_files:
      for embedding_file in embedding_files:
        embeddings_name = embedding_file
        this_vocab_map = vocabs[vocabs_index[embeddings_name]]
        with open(embedding_file, 'r') as f:
          for line in f:
            line = line.strip()
            if line:
              split_line = line.split()
              datum = split_line[0]
              this_vocab_map[datum] = 1

    # build reverse_maps, joint_label_lookup_maps
    for v in vocabs_index.keys():

      this_counts_map = vocabs[vocabs_index[v]]

      # build reverse_lookup map, from int -> string
      this_map = dict(zip(this_counts_map.keys(), range(len(this_counts_map.keys()))))
      tf.logging.debug("vocab %s: %s" % (v, str(list(this_map.items())[:10])))
      reverse_map = dict(zip(range(len(this_counts_map.keys())), this_counts_map.keys()))
      self.oovs[v] = False
      if 'oov' in combined_data_config[v] and combined_data_config[v]['oov']:
        self.oovs[v] = True
      self.reverse_maps[v] = reverse_map
      self.vocab_maps[v] = this_map

      # check whether we need to build joint_label_lookup_map
      if 'label_components' in combined_data_config[v]:
        joint_vocab_map = vocabs[vocabs_index[v]]
        label_components = combined_data_config[v]['label_components']
        component_keys = [vocabs[vocabs_index[d]].keys() for d in label_components]
        component_maps = [dict(zip(comp_keys, range(len(comp_keys)))) for comp_keys in component_keys]
        map_names = ["%s_to_%s" % (v, label_comp) for label_comp in label_components]
        joint_to_comp_maps = [np.zeros([len(joint_vocab_map), 1], dtype=np.int32) for _ in label_components]
        for joint_idx, joint_label in enumerate(joint_vocab_map.keys()):
          split_label = joint_label.split(constants.JOINT_LABEL_SEP)
          for label_comp, comp_map, joint_to_comp_map in zip(split_label, component_maps, joint_to_comp_maps):
            comp_idx = comp_map[label_comp]
            joint_to_comp_map[joint_idx] = comp_idx

        # add them to the master map
        for map_name, joint_to_comp_map in zip(map_names, joint_to_comp_maps):
          self.joint_label_lookup_maps[map_name] = joint_to_comp_map

    for d in vocabs_index.keys():
      this_vocab_map = vocabs[vocabs_index[d]]
      d_clean = util.clean_filename(d)
      with open("%s/%s.txt" % (self.vocabs_dir, d_clean), 'w') as f:
        for k, v in this_vocab_map.items():
          print("%s\t%d" % (k, v), file=f)

    return {k: len(vocabs[vocabs_index[k]]) for k in vocabs_index.keys()}, combined_data_config

  def make_vocab_files(self, data_config, filenames=None, embedding_files=None):
    return self.create_load_or_update_vocab_files(data_config, filenames, embedding_files, False)

  def update(self, filenames=None, embedding_files=None):
    vocab_names_sizes, _ = self.create_load_or_update_vocab_files(self.data_configs, filenames, embedding_files, True)

    # merge new and old
    for vocab_name, vocab_size in vocab_names_sizes.items():
      self.vocab_names_sizes[vocab_name] = vocab_size