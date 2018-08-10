import tensorflow as tf
import value_fns


class ValueFnTests(tf.test.TestCase):

  def test_label_attention_fn(self):

    with self.test_session():

      mode = tf.estimator.ModeKeys.TRAIN

      # num_labels x label_embedding_dim
      label_embeddings = tf.constant([[0.1, 0.1, 0.1, 0.1],
                                      [0.3, 0.3, 0.3, 0.3],
                                      [0.5, 0.5, 0.5, 0.5],
                                      [10, 10, 10, 10],
                                      [1.0, 1.0, 1.0, 1.0]])

      # batch_size x batch_seq_len x num_labels
      label_scores = tf.constant([[[-10., 10., -10., -10., -10.],
                                   [-10., -10., -10., 10., -10.],
                                   [10., 10., 10., -10., -10.]],
                                  [[-10., -10., 10., -10., -10.],
                                   [10., -10., -10., -10., -10.],
                                   [-10., 10., -10., -10., -10.]]])

      # batch_size x batch_seq_len x label_embedding_dim
      expected = tf.constant([[[0.3, 0.3, 0.3, 0.3],
                               [10., 10., 10., 10.],
                               [0.3, 0.3, 0.3, 0.3]],
                              [[0.5, 0.5, 0.5, 0.5],
                               [0.1, 0.1, 0.1, 0.1],
                               [0.3, 0.3, 0.3, 0.3]]])

      result = value_fns.label_attention(mode, label_scores, label_scores, label_embeddings)

      self.assertAllCloseAccordingToType(result.eval(), expected)


if __name__ == '__main__':
  tf.test.main()
