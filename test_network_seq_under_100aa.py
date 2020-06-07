import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from network_v2 import ResNet
from readData_from_TFRec import parse_dataset
from tqdm import tqdm
from utils import expand_dim, calc_pairwise_distances, load_npy_binary, to_distogram, output_to_distancemaps
from utils import pad_mask, pad_primary, pad_tertiary, masked_categorical_cross_entropy, widen_seq


def main():
    """
    paths = []
    for i in range(1, 136):
        paths.append('P:/casp7/casp7/training/100/' + str(i))
    X, mask, y = gather_data_seq_under_limit(paths, 64)
    """

    X = tf.convert_to_tensor(load_npy_binary(path='P:/casp7/casp7/seq_equal64.npy'))
    mask = tf.convert_to_tensor(load_npy_binary(path='P:/casp7/casp7/mask_equal64.npy'))
    y = tf.convert_to_tensor(load_npy_binary(path='P:/casp7/casp7/tertiary_equal64.npy'))

    mask = K.expand_dims(mask, axis=3)
    mask = K.repeat_elements(mask, y.shape[3], axis=3)
    mask = tf.transpose(mask, perm=(0, 3, 1, 2))

    print('Shape of input data: ' + str(X.shape))
    print('Shape of mask ' + str(mask.shape))
    print('Shape of ground truth: ' + str(y.shape))

    callback_es = tf.keras.callbacks.EarlyStopping('loss', verbose=1, patience=10)
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau('loss', verbose=1, patience=5)
    model = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                   learning_rate=0.003, kernel_size=3, batch_size=2, crop_size=64, non_linearity='elu', padding='same',
                   dropout_rate=0.0, training=True)
    """
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True,
                              rankdir='TB', expand_nested=False, dpi=96)
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=model.learning_rate),
                  loss=masked_categorical_cross_entropy())
    model_hist = model.fit([X[0:32], mask[0:32]], y[0:32], batch_size=2, epochs=250, validation_split=0.2,
                           callbacks=[callback_lr, callback_es])
    print(model_hist.history)
    model.save_weights('P:/proteinfolding_alphafold/new_models/weights')
    
    K.clear_session()
    model = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                   learning_rate=0.003, kernel_size=3, batch_size=2, crop_size=64, non_linearity='elu', padding='same',
                   dropout_rate=0.0, training=True)
    model.load_weights('P:/proteinfolding_alphafold/new_models/weights')

    out = model.predict(X)
    distance_maps = output_to_distancemaps(out, 2, 22, 64)
    ground_truth = output_to_distancemaps(y, 2, 22, 64)

    plt.figure()
    plt.subplot(231)
    plt.title("Prediction")
    plt.imshow(distance_maps[0], cmap='viridis_r')
    plt.subplot(232)
    plt.title("Prediction")
    plt.imshow(distance_maps[1], cmap='viridis_r')
    plt.subplot(233)
    plt.title("Prediction")
    plt.imshow(distance_maps[2], cmap='viridis_r')
    plt.subplot(234)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[0], cmap='viridis_r')
    plt.subplot(235)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[1], cmap='viridis_r')
    plt.subplot(236)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[2], cmap='viridis_r')
    plt.show()


def gather_data_seq_under_limit(paths, seq_limit):
    primary_list = []
    tertiary_list = []
    mask_list = []
    desired_shape = (seq_limit, seq_limit)
    for path in paths:
        for primary, evolutionary, tertiary, ter_mask in tqdm(parse_dataset(path)):
            if primary.shape[0] == desired_shape[0]:
                primary = pad_primary(primary, shape=(desired_shape[0]))
                wide_seq = K.cast_to_floatx(widen_seq(primary))
                primary_list.append(wide_seq)
                tertiary = calc_pairwise_distances(tertiary)
                tertiary = pad_tertiary(tertiary, desired_shape)
                tertiary = to_distogram(tertiary, 2, 22, num_bins=64)
                tertiary_list.append(tertiary)
                mask = pad_mask(ter_mask, desired_shape)
                mask_list.append(mask)

    batch_primary = expand_dim(primary_list)
    batch_tertiary = expand_dim(tertiary_list)
    batch_mask = expand_dim(mask_list)

    np.save('P:/casp7/casp7/seq_equal' + str(desired_shape[0]), batch_primary.numpy())
    np.save('P:/casp7/casp7/tertiary_equal' + str(desired_shape[0]), batch_tertiary.numpy())
    np.save('P:/casp7/casp7/mask_equal' + str(desired_shape[0]), batch_mask.numpy())

    return batch_primary, batch_mask, batch_tertiary


if __name__ == '__main__':
    main()
