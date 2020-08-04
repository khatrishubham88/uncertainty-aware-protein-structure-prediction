import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops import array_ops

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

