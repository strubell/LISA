import tensorflow as tf
import nn_utils

'''
Much of this code is adapted from the Tensor2Tensor Transformer implementation:
    https://github.com/tensorflow/tensor2tensor
'''


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      tf.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def attention_bias_ignore_padding(tokens_to_keep):
  """Create a bias tensor to be added to attention logits.
  Args:
    tokens_to_keep: an int Tensor with shape [batch, batch_seq_len].
  Returns:
    A `Tensor` with shape [batch, 1, 1, batch_seq_len].
  """
  # mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
  mask = tf.cast(1 - tokens_to_keep, tf.float32) * -1e9
  return tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     dropout):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope("conv_hidden_relu", [inputs]):
    inputs = tf.expand_dims(inputs, 1)
    in_size = inputs.get_shape().as_list()[-1]
    params1 = tf.get_variable("ff1", [1, 1, in_size, hidden_size])
    params2 = tf.get_variable("ff2", [1, 1, hidden_size, hidden_size])
    params3 = tf.get_variable("ff3", [1, 1, hidden_size, output_size])
    h = tf.nn.conv2d(inputs, params1, [1, 1, 1, 1], "SAME")
    h = nn_utils.leaky_relu(h)
    h = tf.nn.dropout(h, dropout)
    h = tf.nn.conv2d(h, params2, [1, 1, 1, 1], "SAME")
    h = nn_utils.leaky_relu(h)
    h = tf.nn.dropout(h, dropout)
    ret = tf.nn.conv2d(h, params3, [1, 1, 1, 1], "SAME")
    ret = tf.squeeze(ret, 1)
    return ret


def dot_product_attention(q, k, v,
                          bias,
                          special_attention,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope("dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    # concat special_attention to end of logits
    if special_attention:
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), special_attention)), axis=1)

    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, -1)
    weights_drop = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights_drop, v), logits


def compute_qkv(antecedent, input_depth, total_key_depth, total_value_depth):
  """Computes query, key and value.
  Args:
    total_key_depth: num_heads * key_dim
    total_value_depth: num_heads * value_dim
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  params = tf.get_variable("qkv_transform", [1, 1, input_depth, 2*total_key_depth + total_value_depth])
  antecedent = tf.expand_dims(antecedent, 1)
  qkv_combined = tf.nn.conv2d(antecedent, params, [1, 1, 1, 1], "SAME")
  qkv_combined = tf.squeeze(qkv_combined, 1)
  q, k, v = tf.split(qkv_combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
  return q, k, v


def multihead_attention(antecedent,
                        bias,
                        num_heads,
                        head_size,
                        dropout_rate,
                        special_attention,
                        special_values):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  # if total_key_depth % num_heads != 0:
  #   raise ValueError("Key depth (%d) must be divisible by the number of "
  #                    "attention heads (%d)." % (total_key_depth, num_heads))
  # if total_value_depth % num_heads != 0:
  #   raise ValueError("Value depth (%d) must be divisible by the number of "
  #                    "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope("multihead_attention", values=[antecedent]):

    input_size = antecedent.get_shape()[-1]

    total_output_size = head_size * num_heads

    num_basic_attention_heads = num_heads - len(special_attention)
    num_basic_value_heads = num_heads - len(special_values)

    total_basic_key_size = num_basic_attention_heads * head_size
    total_basic_value_size = num_basic_value_heads * head_size

    q, k, v = compute_qkv(antecedent, input_size, total_basic_key_size, total_basic_value_size)
    q = split_heads(q, num_basic_attention_heads)
    k = split_heads(k, num_basic_attention_heads)
    v = split_heads(v, num_basic_value_heads)

    # concat special_values to beginning of values; first k attention heads
    # will attend to k special values
    special_values = list(map(lambda x: tf.expand_dims(x, 1), special_values))
    v = tf.concat(special_values + [v], axis=1)

    # key_depth_per_head = total_key_depth // num_heads
    q *= head_size**-0.5
    x, attn_weights = dot_product_attention(q, k, v, bias, special_attention, dropout_rate)
    x = combine_heads(x)
    params = tf.get_variable("final_proj", [1, 1, total_output_size, total_output_size])
    x = tf.expand_dims(x, 1)
    x = tf.nn.conv2d(x, params, [1, 1, 1, 1], "SAME")
    x = tf.squeeze(x, 1)
    return x, attn_weights


def transformer(inputs, seq_lengths, head_size, num_heads, attn_dropout, relu_dropout, prepost_dropout,
                relu_hidden_size, special_attention, special_values):

  # todo deal with special_attention, special_values

  with tf.name_scope('transformer_layer'):
    mask = attention_bias_ignore_padding(seq_lengths)

    with tf.variable_scope("self_attention"):
      x = nn_utils.layer_norm(inputs)
      y, attn_weights = multihead_attention(x, mask, num_heads, head_size, attn_dropout, special_attention,
                                            special_values)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    with tf.variable_scope("ffnn"):
      x = nn_utils.layer_norm(x)
      y = conv_hidden_relu(x, relu_hidden_size, num_heads * head_size, relu_dropout)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    return x