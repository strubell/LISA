import tensorflow as tf
import constants
import tf_utils
import bert.modeling


def get_bert_mask(bpe_sentences, ids_to_mask=None):
  bert_keep_mask = tf.greater(bpe_sentences, 0)
  if ids_to_mask:
    for idx_to_mask in ids_to_mask:
      bert_keep_mask *= tf.cast(tf.not_equal(bpe_sentences, idx_to_mask), tf.int32)
  return bert_keep_mask


def get_bert_embeddings(bert_dir, bpe_sentences):

  # todo maybe hardcode these paths less
  bert_checkpoint = bert_dir + "/bert_model.ckpt"
  bert_no_pad_mask = get_bert_mask(bpe_sentences)

  bert_config = bert.modeling.BertConfig.from_json_file(bert_dir + "/bert_config.json")
  bert_model = bert.modeling.BertModel(config=bert_config,
                                       is_training=False,
                                       input_ids=tf.cast(bpe_sentences, tf.int32),
                                       input_mask=bert_no_pad_mask)

  tvars = tf.trainable_variables()
  current_variable_scope = tf.get_variable_scope().name
  bert_vars = tf.trainable_variables(scope='%s/bert' % current_variable_scope)

  assignment_map, initialized_variable_names = tf_utils.get_assignment_map_from_checkpoint(tvars, bert_checkpoint)

  tf.train.init_from_checkpoint(bert_checkpoint, {'bert/': '%s/bert/' % current_variable_scope})

  tf.logging.debug("**** BERT Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.debug("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)

  # list of [batch_size, bpe_seq_length, hidden_size]
  bert_embeddings = bert_model.get_all_encoder_layers()
  return bert_embeddings, bert_vars


def get_weighted_avg(bert_vocab, bert_embeddings, bpe_sentences, bpe_lens, l2_penalty=0.001):

  bert_mask_indices = [bert_vocab[s] for s in constants.BERT_MASK_STRS]
  bert_keep_mask = get_bert_mask(bpe_sentences, bert_mask_indices)

  # todo take last k, rather than all layers
  num_bert_layers = len(bert_embeddings)
  bert_layer_weights = tf.get_variable("bert_layer_weights", shape=[num_bert_layers], initializer=tf.zeros_initializer)
  bert_layer_weights_normed = tf.split(tf.nn.softmax(bert_layer_weights + 1.0 / num_bert_layers), num_bert_layers)
  for bert_layer_idx, (bert_layer_weight, bert_layer_output) in enumerate(
      zip(bert_layer_weights_normed, bert_embeddings)):
    bert_embeddings[bert_layer_idx] = tf.expand_dims(bert_layer_output * bert_layer_weight, -1)

  l2_regularizer = tf.contrib.layers.l2_regularizer(l2_penalty)
  bert_weights_l2_loss = l2_regularizer(bert_layer_weights)

  bert_embeddings_concat = tf.concat(bert_embeddings, axis=-1)
  bert_embeddings_avg = tf.reduce_sum(bert_embeddings_concat, -1)

  # [bpe_toks_in_batch x bert_dim]
  bert_embeddings_avg_gather = tf.gather_nd(bert_embeddings_avg, tf.where(bert_keep_mask))

  # use bpe lens to combine bpe reps back into token reps
  bert_dim = bert_embeddings_avg.get_shape().as_list()[-1]

  # [batch_size*batch_seq_len]
  bpe_lens_flat = tf.reshape(bpe_lens, [-1])

  max_bpe_len = tf.reduce_max(bpe_lens)

  # [batch_size*batch_seq_len x max_bpe_len]
  # the number of 1s in scatter_mask (and therefore number of scatter indices) should equal the number of bpe
  # tokens in the batch (and therefore the number of elements in bert_embeddings_avg_gather)
  scatter_mask = tf.sequence_mask(bpe_lens_flat)
  scatter_indices = tf.where(scatter_mask)

  token_batch_shape = tf.shape(bpe_lens)
  batch_size = token_batch_shape[0]
  batch_seq_len = token_batch_shape[1]

  bert_reps_scatter = tf.scatter_nd(scatter_indices, bert_embeddings_avg_gather,
                                    [batch_size * batch_seq_len, max_bpe_len, bert_dim])

  # average over bpes to get tokens
  bert_tokens = tf.reshape(tf.reduce_mean(bert_reps_scatter, axis=1), [batch_size, batch_seq_len, bert_dim])

  return bert_tokens, bert_weights_l2_loss