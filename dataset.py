import tensorflow as tf
from data_generator import conll_data_generator
import constants


def map_strings_to_ints(vocab_lookup_ops, data_config, data_names):
  def _mapper(d):
    intmapped = []
    for i, datum_name in enumerate(data_names):
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?
        if 'type' in data_config[datum_name] and data_config[datum_name]['type'] == 'range':
          idx = data_config[datum_name]['conll_idx']
          last_idx = i + idx[1] if idx[1] > 0 else -1
          intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:last_idx]))
        else:
          intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i]), -1))
      else:
        intmapped.append(tf.expand_dims(tf.string_to_number(d[:, i], out_type=tf.int64), -1))


    # this is where the order of features/labels in input gets defined
    return tf.cast(tf.concat(intmapped, axis=-1), tf.int32) # todo: can i have these just naturally be int32?

  return _mapper


'''
TODO: comment
'''
def get_data_iterator(data_filename, data_config, vocab_lookup_ops, batch_size, num_epochs, is_train):
  with tf.device('/cpu:0'):

    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    feature_label_names = [d for d in data_config.keys() if \
                           ('feature' in data_config[d] and data_config[d]['feature']) or \
                           ('label' in data_config[d] and data_config[d]['label'])]

    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filename, data_config),
                                             output_shapes=[None, None], output_types=tf.string)


    # intmap the dataset
    dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names), num_parallel_calls=8)

    # do batching
    dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d: tf.shape(d)[0],
                                                                      bucket_boundaries=[20, 30, 50, 80],  # todo: optimal?
                                                                      bucket_batch_sizes=[batch_size] * 5,
                                                                      padded_shapes=dataset.output_shapes,
                                                                      padding_values = constants.PAD_VALUE))
                                                                      # padding_values=tf.constant(constants.PAD_VALUE,
                                                                      #                            dtype=tf.int64)))
    dataset = dataset.cache()

    # shuffle and expand out epochs if training
    if is_train:
      dataset = dataset.repeat(num_epochs)
      dataset = dataset.shuffle(batch_size * 100)
      # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size, count=num_epochs))

    # todo should the buffer be bigger?
    dataset.prefetch(buffer_size=1)

    # create the iterator
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    # features, labels = iterator.get_next()
    # return features, labels
    return iterator.get_next()
