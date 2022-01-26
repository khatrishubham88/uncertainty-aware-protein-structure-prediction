import math
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from utils import prob_to_class


def save_tfrecord(y, fname):
    serialized_tensor = tf.io.serialize_tensor(y)
    tf.io.write_file(fname+".tfrecord", serialized_tensor)


def read_tfrecord(fname):
    read_data = tf.io.read_file(fname+".tfrecord")
    return tf.io.parse_tensor(read_data, tf.float32)

def sample_misspecification(y_mean, y_pred, mask=None):
    y_mean = tf.convert_to_tensor(y_mean, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    diff = y_mean - y_pred
    diff = tf.math.reduce_sum(diff, axis=-1)
    if mask is not None:
        diff = diff*mask
    diff = tf.math.square(diff)
    total = tf.math.reduce_sum(diff)
    if mask is not None:
        num_elems = tf.math.reduce_sum(mask)
    else:
        num_elems = tf.math.reduce_prod(array_ops.shape(y_mean))
    if tf.cast(num_elems, dtype=tf.float32)>tf.cast(tf.math.reduce_prod(array_ops.shape(y_mean)), dtype=tf.float32):
        raise ValueError("Incorrect result from elems")
    if num_elems<1:
        total = 0.0
    return total, num_elems

def total_model_noise(y_true, y_pred, num_classes, mask=None):
    pred_classes = prob_to_class(y_pred, num_classes)
    model_noise, count = sample_misspecification(pred_classes, y_true, mask)
    if count !=0:
        model_noise /= float(count)
    else:
        model_noise = 0.0
    try:
        model_noise = tf.math.sqrt(tf.cast(model_noise, dtype=tf.float32))
    except:
        model_noise = math.sqrt(float(model_noise))
    return model_noise
