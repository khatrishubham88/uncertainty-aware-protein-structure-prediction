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


def sample_misspecification(y_mean, y_pred):
    y_mean = tf.convert_to_tensor(y_mean, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    diff = y_mean - y_pred
    diff = tf.math.square(diff)
    total = tf.math.reduce_sum(diff)
    num_elems = tf.math.reduce_prod(array_ops.shape(y_mean))
    return total, num_elems


def total_model_noise(y_true, y_pred, num_classes):
    pred_classes = prob_to_class(y_pred, num_classes)
    model_noise, count = sample_misspecification(pred_classes, y_true)
    model_noise /= float(count)
    try:
        model_noise = tf.math.sqrt(tf.cast(model_noise))
    except:
        model_noise = math.sqrt(float(model_noise))
    return model_noise
