import tensorflow as tf


def leaky_relu(x): return tf.maximum(0.1 * x, x)


def set_vars_to_moving_average(moving_averager):
  moving_avg_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
  return tf.group(*[tf.assign(x, moving_averager.average(x)) for x in moving_avg_variables])


def layer_norm(inputs, epsilon=1e-6):
  """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope("layer_norm"):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
    outputs = gamma * normalized + beta
  return outputs


# def orthonormal_initializer(input_size, output_size):
#   """"""
#
#   if not tf.get_variable_scope().reuse:
#     print(tf.get_variable_scope().name)
#     I = np.eye(output_size)
#     lr = .1
#     eps = .05 / (output_size + input_size)
#     success = False
#     while not success:
#       Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#       for i in range(100):
#         QTQmI = Q.T.dot(Q) - I
#         loss = np.sum(QTQmI ** 2 / 2)
#         Q2 = Q ** 2
#         Q -= lr * Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
#         if np.isnan(Q[0, 0]):
#           lr /= 2
#           break
#       if np.isfinite(loss) and np.max(Q) < 1e6:
#         success = True
#       eps *= 2
#     print('Orthogonal pretrainer loss: %.2e' % loss)
#   else:
#     print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
#     Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#   return Q.astype(np.float32)


def linear_layer(inputs, output_size, add_bias=True, n_splits=1, initializer=None):
  """"""

  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  output_size *= n_splits

  with tf.variable_scope('Linear'):
    # Reformat the input
    total_input_size = 0
    shapes = [a.get_shape().as_list() for a in inputs]
    for shape in shapes:
      total_input_size += shape[-1]
    input_shape = tf.shape(inputs[0])
    output_shape = []
    for i in range(len(shapes[0])):
      output_shape.append(input_shape[i])
    output_shape[-1] = output_size
    output_shape = tf.stack(output_shape)
    for i, (input_, shape) in enumerate(zip(inputs, shapes)):
      inputs[i] = tf.reshape(input_, [-1, shape[-1]])
    concatenation = tf.concat(axis=1, values=inputs)

    # Get the matrix
    if initializer is None:
      initializer = tf.initializers.orthogonal
      # mat = orthonormal_initializer(total_input_size, output_size // n_splits)
      # mat = np.concatenate([mat] * n_splits, axis=1)
      # initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
    # tf.add_to_collection('Weights', matrix)

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
    else:
      bias = 0

    # Do the multiplication
    new = tf.matmul(concatenation, matrix) + bias
    new = tf.reshape(new, output_shape)
    new.set_shape([tf.Dimension(None) for _ in range(len(shapes[0]) - 1)] + [tf.Dimension(output_size)])
    if n_splits > 1:
      return tf.split(axis=len(new.get_shape().as_list()) - 1, num_or_size_splits=n_splits, value=new)
    else:
      return new


# TODO clean this up
def MLP(inputs, output_size, func=leaky_relu, keep_prob=1.0, n_splits=1):
  """"""

  input_shape = inputs.get_shape().as_list()
  n_dims = len(input_shape)
  batch_size = tf.shape(inputs)[0]
  input_size = input_shape[-1]
  shape_to_set = [tf.Dimension(None)] * (n_dims - 1) + [tf.Dimension(output_size)]

  if keep_prob < 1:
    noise_shape = tf.stack([batch_size] + [1] * (n_dims - 2) + [input_size])
    inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)

  linear = linear_layer(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=True)
  if n_splits == 1:
    linear = [linear]
  for i, split in enumerate(linear):
    split = func(split)
    split.set_shape(shape_to_set)
    linear[i] = split
  if n_splits == 1:
    return linear[0]
  else:
    return linear


def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None):
  """"""

  with tf.variable_scope('Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims - 2]
    inputs1_size = inputs1.get_shape().as_list()[-1]

    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims - 2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    # output_shape = []
    batch_size1 = 1
    batch_size2 = 1
    for i in range(ndims - 2):
      batch_size1 *= inputs1_shape[i]
      batch_size2 *= inputs2_shape[i]
      # output_shape.append(inputs1_shape[i])
    # output_shape.append(inputs1_bucket_size)
    # output_shape.append(output_size)
    # output_shape.append(inputs2_bucket_size)
    # output_shape = tf.stack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.stack([batch_size1, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.stack([batch_size2, inputs2_bucket_size, inputs2_size]))
    if add_bias1:
      inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, inputs1_bucket_size, 1]))])
    if add_bias2:
      inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, inputs2_bucket_size, 1]))])

    # Get the matrix
    if initializer is None:
      # mat = orthonormal_initializer(inputs1_size + add_bias1, inputs2_size + add_bias2)[:, None, :]
      # mat = np.concatenate([mat] * output_size, axis=1)
      # initializer = tf.constant_initializer(mat)
      initializer = tf.initializers.orthogonal
    weights = tf.get_variable('Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                              initializer=initializer)
    # tf.add_to_collection('Weights', weights)

    # inputs1: num_triggers_in_batch x 1 x self.trigger_mlp_size
    # inputs2: batch x seq_len x self.role_mlp_size

    # Do the multiplications
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + add_bias1]), tf.reshape(weights, [inputs1_size + add_bias1, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    lin_reshape = tf.reshape(lin, tf.stack([batch_size1, inputs1_bucket_size * output_size, inputs2_size + add_bias2]))
    bilin = tf.matmul(lin_reshape, inputs2, adjoint_b=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
      bilin += tf.expand_dims(bias, 1)

    return bilin


def bilinear_classifier(inputs1, inputs2, keep_prob, add_bias1=True, add_bias2=False):
  """"""

  input_shape = tf.shape(inputs1)
  batch_size = input_shape[0]
  bucket_size = input_shape[1]
  input_size = inputs1.get_shape().as_list()[-1]

  if keep_prob < 1:
    noise_shape = [batch_size, 1, input_size]
    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

  bilin = bilinear(inputs1, inputs2, 1,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())
  output = tf.reshape(bilin, [batch_size, bucket_size, bucket_size])
  # output = tf.squeeze(bilin)
  return output


def bilinear_classifier_nary(inputs1, inputs2, n_classes, keep_prob, add_bias1=True, add_bias2=True):
  """"""

  input_shape1 = tf.shape(inputs1)
  input_shape2 = tf.shape(inputs2)

  batch_size1 = input_shape1[0]
  batch_size2 = input_shape2[0]

  # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
  bucket_size1 = input_shape1[1]
  bucket_size2 = input_shape2[1]
  input_size1 = inputs1.get_shape().as_list()[-1]
  input_size2 = inputs2.get_shape().as_list()[-1]

  input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
  input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

  if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
    noise_shape1 = tf.stack([batch_size1, 1, input_size1])
    noise_shape2 = tf.stack([batch_size2, 1, input_size2])

    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
  inputs1.set_shape(input_shape_to_set1)
  inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, 1]))])
  inputs2.set_shape(input_shape_to_set2)

  bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())

  return bilin


def conditional_bilinear_classifier(inputs1, inputs2, n_classes, probs, keep_prob, add_bias1=True, add_bias2=True):
  """"""

  input_shape = tf.shape(inputs1)
  batch_size = input_shape[0]
  bucket_size = input_shape[1]
  input_size = inputs1.get_shape().as_list()[-1]
  input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size + 1]
  # output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
  if len(probs.get_shape().as_list()) == 2:
    probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
  else:
    probs = tf.stop_gradient(probs)

  if keep_prob < 1:
    noise_shape = tf.stack([batch_size, 1, input_size])
    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
  inputs1.set_shape(input_shape_to_set)
  inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
  inputs2.set_shape(input_shape_to_set)

  bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())
  bilin = tf.reshape(bilin, [batch_size, bucket_size, n_classes, bucket_size])
  weighted_bilin = tf.squeeze(tf.matmul(bilin, tf.expand_dims(probs, 3)), -1)

  return weighted_bilin, bilin