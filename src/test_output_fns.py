import tensorflow as tf
import numpy as np
import output_fns


class OutputFnTests(tf.test.TestCase):

  # correct solution:
  def softmax(self, x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

  def test_get_separate_scores_preds_from_joint(self):

    with self.test_session():

      joint_num_labels = 6

      # joint_maps = {
      #   'joint_to_single1':
      #     tf.constant([[0, 1],
      #                  [1, 1],
      #                  [2, 1],
      #                  [3, 0],
      #                  [4, 0],
      #                  [5, 0]]),
      #   'joint_to_single2':
      #     tf.constant([[0, 0],
      #                  [1, 1],
      #                  [2, 2],
      #                  [3, 0],
      #                  [4, 1],
      #                  [5, 2]])
      # }
      joint_maps = {
        'joint_to_single1':
          tf.constant([[0],
                       [0],
                       [0],
                       [1],
                       [1],
                       [1]]),
        'joint_to_single2':
          tf.constant([[0],
                       [1],
                       [2],
                       [0],
                       [1],
                       [2]])
      }
      # joint_maps = {
      #   'joint_to_single1':
      #     tf.constant([1, 1, 1, 0, 0, 0]),
      #   'joint_to_single2':
      #     tf.constant([0, 1, 2, 0, 1, 2])
      # }

      # joint_predictions: batch_size x batch_seq_len
      joint_predictions = tf.constant([[1, 3, 0],
                                       [2, 0, 1]])

      # joint_scores: batch_size x batch_seq_len x num_joint_labels
      joint_scores = tf.constant([[[-10., 10., -10., -10., -10., -10.],
                                   [-10., -10., -10., 10., -10., -10.],
                                   [10., 9., 9., -10., -10., -10.]],
                                  [[-10., -10., 10., -10., -10., -10.],
                                   [10., -10., 10., -10., -10., 9.],
                                   [-10., 10., -10., -10., -10., 9.]]])

      joint_probabilities = [[[2.0611536e-09, 9.9999999e-01, 2.0611536e-09,
                               2.0611536e-09, 2.0611536e-09, 2.0611536e-09],
                              [2.0611536e-09, 2.0611536e-09, 2.0611536e-09,
                               9.9999999e-01, 2.0611536e-09, 2.0611536e-09],
                              [5.76116883e-01, 2.11941557e-01, 2.11941557e-01,
                               1.18746540e-09, 1.18746540e-09, 1.18746540e-09]],

                             [[2.0611536e-09, 2.0611536e-09, 9.9999999e-01,
                               2.0611536e-09, 2.0611536e-09, 2.0611536e-09],
                              [4.22318797e-01, 8.70463919e-10, 4.22318797e-01,
                               8.70463919e-10, 8.70463919e-10, 1.55362403e-01],
                              [1.50682403e-09, 7.31058574e-01, 1.50682403e-09,
                               1.50682403e-09, 1.50682403e-09, 2.68941420e-01]]]

      single1_predictions_expected = tf.constant([[0, 1, 0],
                                                  [0, 0, 0]])

      # These predictions come from joint_scores rather than single task scores, which
      # have a tie and default to the smaller-index label. This is why they are different
      # than if we derived them from the single task scores, which do not have a tie.
      single2_predictions_expected = tf.constant([[1, 0, 0],
                                                  [2, 0, 1]])

      single1_probabilities_expected = np.array([[[0.9999999941223071, 6.1834607999999995e-09],
                                         [6.1834607999999995e-09, 0.9999999941223071],
                                         [0.999999997, 3.5623962e-09]],
                                        [[0.9999999941223071, 6.1834607999999995e-09],
                                         [0.844637594870464, 0.15536240474092786],
                                         [0.731058577013648, 0.26894142301364804]]])

      single2_probabilities_expected = np.array([[[4.1223072e-09, 0.9999999920611535, 4.1223072e-09],
                                         [0.9999999920611535, 4.1223072e-09, 4.1223072e-09],
                                         [0.5761168841874653, 0.2119415581874654, 0.2119415581874654]],
                                        [[4.1223072e-09, 4.1223072e-09, 0.9999999920611535],
                                         [0.42231879787046395, 1.740927838e-09, 0.5776812],
                                         [2.06115362e-09, 0.731058575506824, 0.26894142150682404]]])

      # joint_outputs: a dict containing joint 'predictions' and 'scores'
      joint_outputs = {
        'predictions': joint_predictions,
        'scores': joint_scores
      }

      result = output_fns.get_separate_scores_preds_from_joint(joint_outputs, joint_maps, joint_num_labels)

      # result should contain an entry for task_predictions and task_scores for each task
      self.assertAllCloseAccordingToType(result['single1_predictions'].eval(), single1_predictions_expected)
      self.assertAllCloseAccordingToType(result['single2_predictions'].eval(), single2_predictions_expected)
      self.assertAllCloseAccordingToType(result['single1_probabilities'].eval(), single1_probabilities_expected)
      self.assertAllCloseAccordingToType(result['single2_probabilities'].eval(), single2_probabilities_expected)


if __name__ == '__main__':
  tf.test.main()
