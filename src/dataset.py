import tensorflow as tf
import constants
import util
from functools import partial
from data_generator import conll_data_generator


def map_strings_to_ints(vocab_lookup_ops, data_config, idx_map):
  """
  An important glue function that maps the list of converted fields from the data to ints.
  Here we manage the fact that there may be fields which map to variable-length.

  :param vocab_lookup_ops: map of named tf lookup ops that map strings to ints
  :param data_config: data configuration map (loaded from data config json)
  :param idx_map: map from names of features/labels to (start, end) indices in the input, d
  :return: mapping function that takes a list of strings and returns a list of ints
  """
  def _mapper(d):
    intmapped = {}
    # for each feature/label
    for datum_name, datum_idx in idx_map.items():
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?
        if 'type' in data_config[datum_name]:
          if data_config[datum_name]['type'] == 'range':
            if datum_idx[1] == -1:
              # this means just take the rest of the fields
              # todo we need the variable-length feat to come last, gross
              intmapped[datum_name] = vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]:])
            else:
              # this feature/label consists of idx[1]-idx[0] elements
              intmapped[datum_name] = vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]:datum_idx[1]])
        else:
          # simplest case: single element
          intmapped[datum_name] = vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]])
      else:
        # simple case: single element that needs to be converted to an int
        intmapped[datum_name] = tf.string_to_number(d[:, datum_idx[0]], out_type=tf.int64)

    # this is where the order of features/labels in input gets defined
    return intmapped

  return _mapper


def get_dataset(data_filenames, data_config, vocab, batch_size, num_epochs, shuffle, shuffle_buffer_multiplier=1):

  # this needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops()

  bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  # todo do something smarter with multiple files + parallel?

  with tf.device('/cpu:0'):

    all_features = {}
    all_labels = {}

    padding_values = ({}, {})
    pad_constant_tf = tf.constant(constants.PAD_VALUE, dtype=tf.int64)

    # for each chunk of data (set of mappings defined in mapping config, and optional example converter
    for d, this_config in data_config.items():

      mapping_config = this_config["mappings"]

      feature_idx_map = util.load_input_idx_maps(mapping_config, 'feature', ['feature', 'label'])
      label_idx_map = util.load_input_idx_maps(mapping_config, 'label', ['feature', 'label'])

      # get the dataset
      dataset = tf.data.Dataset.from_generator(partial(conll_data_generator,
                                                       filenames=data_filenames,
                                                       data_config=this_config),
                                               output_shapes=[None, None],
                                               output_types=tf.string)

      # intmap the dataset
      if feature_idx_map:
        padding_values[0].update({k: pad_constant_tf for k in feature_idx_map})
        features = dataset.map(map_strings_to_ints(vocab_lookup_ops, mapping_config, feature_idx_map),
                               num_parallel_calls=8)
        all_features[d] = features

      if label_idx_map:
        padding_values[1].update({k: pad_constant_tf for k in label_idx_map})
        labels = dataset.map(map_strings_to_ints(vocab_lookup_ops, mapping_config, label_idx_map), num_parallel_calls=8)
        all_labels[d] = labels

    # need to flatten nested maps for PREDICT mode / model exporting
    # right now all_features is a map containing datasets
    # all_features_list = list(all_features.values())
    # feats_concat = all_features_list.pop()
    # for d in all_features_list:
    #   feats_concat.concatenate(d)

    def flatten_dataset(f, l):
      d0_flat = {}
      d1_flat = {}
      for v in f.values():
        d0_flat.update(v)
      for v in l.values():
        d1_flat.update(v)
      return d0_flat, d1_flat

    # f = tf.data.Dataset.concatenate(tuple())

    dataset = tf.data.Dataset.zip((all_features, all_labels))

    dataset = dataset.map(flatten_dataset)

    dataset = dataset.cache()

    # grab the length of the first dim of the first thing in the first map in features
    def length_func(f, _):
      # return tf.shape(next(iter(next(iter(f.values())).values())))[0]
      return tf.shape((next(iter(f.values()))))[0]

    # do batching
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=length_func,
                                                                      bucket_boundaries=bucket_boundaries,
                                                                      bucket_batch_sizes=bucket_batch_sizes,
                                                                      padded_shapes=dataset.output_shapes,
                                                                      padding_values=padding_values))

    # shuffle and expand out epochs if training
    if shuffle:
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * shuffle_buffer_multiplier,
                                                                 count=num_epochs))

    # todo should the buffer be bigger?
    dataset.prefetch(buffer_size=1)

    return dataset


def get_data_iterator(dataset):
    # create the iterator
    # it has to be initializable due to the lookup tables
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    feats, labels = iterator.get_next()

    return feats, labels

