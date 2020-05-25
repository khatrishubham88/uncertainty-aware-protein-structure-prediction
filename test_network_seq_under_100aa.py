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
    """
    paths = []
    for i in range(1, 136):
        paths.append('/storage/remote/atcremers45/s0237/casp7/training/100/' + str(i))
    X, mask, y = gather_data_seq_under_limit(paths, 64)
    """

    X = load_npy_binary(path='/storage/remote/atcremers45/s0237/casp7/training/seq64.npy')
    mask = load_npy_binary(path='/storage/remote/atcremers45/s0237/casp7/training/mask64.npy')
    y = load_npy_binary(path='/storage/remote/atcremers45/s0237/casp7/training/tertiary64.npy')

    print('Shape of input data: ' + str(X.shape))
    print('Shape of mask ' + str(mask.shape))
    print('Shape of ground truth: ' + str(y.shape))
    print(X[10])

    distance_maps = output_to_distancemaps(y, 2, 22, 64)

    plt.figure()
    plt.subplot(121)
    plt.title("Ground Truth")
    plt.imshow(mask[10], cmap='viridis_r')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(distance_maps[10], cmap='viridis_r')
    plt.colorbar()
    plt.show()

    """
    K.clear_session()
    model = tf.keras.models.load_model('P:/proteinfolding_alphafold/models/test_28_with_64_4.h5', compile=False)

    distance_maps = output_to_distancemaps(y, 2, 22, 64)
    test = model(X[0:64])

    # Instantiate ResNet model

    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=16, crop_size=64, dropout_rate=0.15)
    model = nn.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001),
                  loss=masked_categorical_cross_entropy(mask))
    model_hist = model.fit(X, y, batch_size=16, epochs=100, validation_split=0.2)
    print(model_hist.history)
    model.save("/usr/prakt/s0237/pcss20-proteinfolding/models/tests/seq_equal_64/custom_loss/model_b16_fs.h5")
    
    test = model.predict(X[0:32])
    test = output_to_distancemaps(test, 2, 22, 64)

    plt.figure()
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.imshow(distance_maps[0], cmap='viridis_r')
    plt.subplot(132)
    plt.title("Prediction by model")
    plt.imshow(test[0], cmap='viridis_r')
    plt.subplot(133)
    plt.title("Prediction by model")
    plt.imshow(mask[0], cmap='viridis_r')
    plt.show()

    plt.figure()
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.imshow(distance_maps[10], cmap='viridis_r')
    plt.subplot(132)
    plt.title("Prediction by model")
    plt.imshow(test[10], cmap='viridis_r')
    plt.subplot(133)
    plt.title("Prediction by model")
    plt.imshow(mask[10], cmap='viridis_r')
    plt.show()

    plt.figure()
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.imshow(distance_maps[20], cmap='viridis_r')
    plt.subplot(132)
    plt.title("Prediction by model")
    plt.imshow(test[20], cmap='viridis_r')
    plt.subplot(133)
    plt.title("Prediction by model")
    plt.imshow(mask[20], cmap='viridis_r')
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
