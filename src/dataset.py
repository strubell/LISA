import tensorflow as tf
import constants
import util
from data_generator import conll_data_generator, bert_data_generator


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
    intmapped = []
    # for each feature/label
    for datum_name, datum_idx in idx_map.items():
      # datum_idx = datum_idx[0]
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?
        if 'type' in data_config[datum_name]:
          if data_config[datum_name]['type'] == 'range':
            if datum_idx[1] == -1:
              # this means just take the rest of the fields
              # todo we need the variable-length feat to come last, gross
              intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]:]))
            else:
              # this feature/label consists of idx[1]-idx[0] elements
              intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]:datum_idx[1]]))
        else:
          # simplest case: single element
          intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, datum_idx[0]]), -1))
      else:
        # simple case: single element that needs to be converted to an int
        intmapped.append(tf.expand_dims(tf.string_to_number(d[:, datum_idx[0]], out_type=tf.int64), -1))

    # this is where the order of features/labels in input gets defined
    # todo: can i have these come out of the lookup as int32?
    return tf.cast(tf.concat(intmapped, axis=-1), tf.int32)

  return _mapper

# todo: gross, fix
def map_strings_to_ints_bert(vocab_lookup_ops, data_config, idx_map):
  """
  An important glue function that maps the list of converted fields from the data to ints.
  Here we manage the fact that there may be fields which map to variable-length.

  :param vocab_lookup_ops: map of named tf lookup ops that map strings to ints
  :param data_config: data configuration map (loaded from data config json)
  :param idx_map: map from names of features/labels to (start, end) indices in the input, d
  :return: mapping function that takes a list of strings and returns a list of ints
  """
  def _mapper(d):
    intmapped = []

    # for each feature/label
    for datum_name, datum_idx in idx_map.items():
      # datum_idx = datum_idx[0]
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?

        # simplest case: single element
        intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d), -1))
      else:
        # simple case: single element that needs to be converted to an int
        intmapped.append(tf.expand_dims(tf.string_to_number(d, out_type=tf.int64), -1))

    # this is where the order of features/labels in input gets defined
    # todo: can i have these come out of the lookup as int32?
    return tf.cast(tf.concat(intmapped, axis=-1), tf.int32)

  return _mapper


def get_data_iterator(data_filenames, data_configs, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                      shuffle_buffer_multiplier):

  bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  # todo do something smarter with multiple files + parallel?

  data_config = data_configs[0]
  sentences_config = data_configs[1]

  with tf.device('/cpu:0'):

    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    # feature_label_names = [d for d in data_config.keys() if \
    #                        ('feature' in data_config[d] and data_config[d]['feature']) or
    #                        ('label' in data_config[d] and data_config[d]['label'])]
    # feature_names = [d for d in data_config.keys() if 'feature' in data_config[d] and data_config[d]['feature']]
    # label_names = [d for d in data_config.keys() if 'label' in data_config[d] and data_config[d]['label']]

    feature_idx_map = util.load_input_idx_maps(data_config, 'feature', ['feature', 'label'])
    label_idx_map = util.load_input_idx_maps(data_config, 'label', ['feature', 'label'])
    sent_idx_map = util.load_input_idx_maps(sentences_config, 'feature', ['feature', 'label'])

    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filenames, data_config),
                                             output_shapes=[None, None], output_types=tf.string)

    # intmap the dataset
    features = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_idx_map), num_parallel_calls=8)
    # dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names))

    labels = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, label_idx_map), num_parallel_calls=8)

    sentences = tf.data.Dataset.from_generator(lambda: bert_data_generator(data_filenames, sentences_config),
                                               output_shapes=[None], output_types=tf.string)

    # todo: need to add [CLS] ... [SEP], then remove them
    intmapped_sentences = sentences.map(map_strings_to_ints_bert(vocab_lookup_ops, sentences_config, sent_idx_map), num_parallel_calls=8)

    features = tf.data.Dataset.zip((features, intmapped_sentences))
    dataset = tf.data.Dataset.zip((features, labels))

    dataset = dataset.cache()

    # do batching
    # todo automatically generate the padding
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d, s: tf.shape(d[0])[0],
                                                                      bucket_boundaries=bucket_boundaries,
                                                                      bucket_batch_sizes=bucket_batch_sizes,
                                                                      padded_shapes=dataset.output_shapes,
                                                                      # padding_values=((constants.PAD_VALUE, ''), constants.PAD_VALUE)))
                                                                      padding_values=((constants.PAD_VALUE, constants.PAD_VALUE), constants.PAD_VALUE)))

    # dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda s: tf.shape(s)[0],
    #                                                                   bucket_boundaries=bucket_boundaries,
    #                                                                   bucket_batch_sizes=bucket_batch_sizes,
    #                                                                   padded_shapes=dataset.output_shapes,
    #                                                                   padding_values=('')))

    # shuffle and expand out epochs if training
    if shuffle:
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*shuffle_buffer_multiplier,
                                                                 count=num_epochs))
      # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size,
      #                                                            count=num_epochs))

    # todo should the buffer be bigger?
    dataset.prefetch(buffer_size=1)

    # create the iterator
    # it has to be initializable due to the lookup tables
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    feats, labels = iterator.get_next()

    return {'features': feats[0], 'sentences': feats[1]}, labels
