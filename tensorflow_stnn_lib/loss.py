import tensorflow as tf
import tensorflow.keras.backend as K

def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = 1.0) -> tf.Tensor:
    """Contrastive Loss function

    :param y_true: unused target labels
    :type y_true: tf.Tensor
    :param y_pred: model output (a list with positive and negative distances)
    :type y_pred: tf.Tensor
    :param margin: desired separation margin between positive and negative examples, defaults to 1.0
    :type margin: float, optional
    :return: Contastive loss tensor
    :rtype: tf.Tensor
    """
    y = tf.cast(y, y_pred.dtype)
    squared_y_preds = K.square(y_pred)
    squared_margin = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y * squared_y_preds + (1 - y) * squared_margin)

def triplet_loss(y_true: tf.Tensor, y_pred: tf.Tensor, margin: float = 1.0) -> tf.Tensor:    
    """Triplet Loss function

    :param y_true: unused target labels
    :type y_true: tf.Tensor
    :param y_pred: model output (a list with positive and negative distances)
    :type y_pred: tf.Tensor
    :param margin: desired separation margin between positive and negative examples, defaults to 1.0
    :type margin: float, optional
    :return: triplet loss tensor
    :rtype: tf.Tensor
    """
    pos_distance = y_pred[0]
    neg_distance = y_pred[1]
    return K.maximum(pos_distance**2 - neg_distance**2 + margin, 0.0)
    