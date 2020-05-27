import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from network import ResNet
from readData_from_TFRec import widen_seq, parse_dataset
from tqdm import tqdm
from utils import expand_dim, calc_pairwise_distances, to_distogram, load_npy_binary, output_to_distancemaps
from utils import pad_mask, pad_primary, pad_tertiary, masked_categorical_cross_entropy

def main():
    #paths = []
    #for i in range(1, 136):
    #    paths.append('P:/casp7/casp7/training/100/' + str(i))
    #X, mask, y = gather_data_seq_under_limit(paths, 64)

    X = load_npy_binary(path='P:/casp7/casp7/seq_equal64.npy')
    mask = load_npy_binary(path='P:/casp7/casp7/mask_equal64.npy')
    y = load_npy_binary(path='P:/casp7/casp7/tertiary_equal64.npy')

    print('Shape of input data: ' + str(X.shape))
    print('Shape of mask ' + str(mask.shape))
    print('Shape of ground truth: ' + str(y.shape))

    """
    ground_truth = output_to_distancemaps(y, 2, 22, 64)
    plt.figure()
    plt.subplot(121)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[10], cmap='viridis_r')
    plt.colorbar()
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask[10], cmap='viridis_r')
    plt.colorbar()
    plt.show()
    """

    K.clear_session()
    model = tf.keras.models.load_model(
        'P:/proteinfolding_alphafold/models/tests/seq_equal_64/keras_loss/model_b2_32s.h5', compile=False)

    out = model.predict(X[0:64])
    distance_maps = output_to_distancemaps(out, 2, 22, 64)
    ground_truth = output_to_distancemaps(y, 2, 22, 64)

    plt.figure()
    plt.subplot(231)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[0], cmap='viridis_r')
    plt.subplot(232)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[1], cmap='viridis_r')
    plt.subplot(233)
    plt.title("Ground Truth")
    plt.imshow(ground_truth[2], cmap='viridis_r')
    plt.subplot(234)
    plt.title("Prediction")
    plt.imshow(distance_maps[0], cmap='viridis_r')
    plt.subplot(235)
    plt.title("Prediction")
    plt.imshow(distance_maps[1], cmap='viridis_r')
    plt.subplot(236)
    plt.title("Prediction")
    plt.imshow(distance_maps[2], cmap='viridis_r')
    plt.show()

    """
    K.clear_session()
    model = tf.keras.models.load_model(
        'P:/proteinfolding_alphafold/models/tests/seq_equal_64/keras_loss/model_b16_fs.h5',
        compile=False)

    out = model.predict(X[0:64])
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
    """

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
                tertiary = to_distogram(tertiary, min=2, max=22, num_bins=64)
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
