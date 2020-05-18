import tensorflow as tf
import tensorflow.keras.backend as K


def mask_2d_to_3d(masks_2d):
    mask_3d = K.stack(masks_2d, axis=0)

    return mask_3d


if __name__ == "__main__":

    masks_2d = [K.random_uniform(shape=(64, 64), minval=0, maxval=2, dtype=tf.dtypes.int32),
                K.random_uniform(shape=(64, 64), minval=0, maxval=2, dtype=tf.dtypes.int32),
                K.random_uniform(shape=(64, 64), minval=0, maxval=2, dtype=tf.dtypes.int32)]

    mask_3d = mask_2d_to_3d(masks_2d)
