import tensorflow as tf
import tensorflow.keras.backend as K
from typing import *

def euclidean_distance(vectors: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Euclidean distance between two vectors.
    Ref: https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/

    :param vectors: Tuple with two input tensors
    :type vectors: Tuple[tf.Tensor, tf.Tensor]
    :return: euclidean distance tensor
    :rtype: tf.Tensor
    """
    # unpack the vectors into separate lists
    featsA = vectors[0] 
    featsB = vectors[1]

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def cosine_distance(vectors: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Cosine distance between two vectors.
    
    :param vectors: Tuple with two input tensors
    :type vectors: Tuple[tf.Tensor, tf.Tensor]
    :return: cosine distance tensor
    :rtype: tf.Tensor
    """
    # unpack the vectors into separate lists
    featsA = vectors[0] 
    featsB = vectors[1]

    dot_prod = tf.reduce_sum(featsA*featsB, axis=1, keepdims=True)
    normA = tf.norm(featsA, axis=1, keepdims=True)
    normB = tf.norm(featsA, axis=1, keepdims=True)
    cos_sim = dot_prod/K.maximum(normA*normB, K.epsilon())
    return 1 - cos_sim