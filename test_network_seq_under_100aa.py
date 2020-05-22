import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from readData_from_TFRec import widen_seq, parse_dataset
from tqdm import tqdm
from utils import mask_2d_to_3d, primary_2d_to_3d, pad_tensor, calc_pairwise_distances


def gather_data_seq_under_limit(paths):
    primary_list = []
    tertiary_list = []
    mask_list = []
    lengths = []
    max_seq_length = 64

    for path in paths:
        for primary, evolutionary, tertiary, ter_mask in tqdm(parse_dataset(path)):
            if primary.shape[0] <= max_seq_length:
                wide_seq = K.cast_to_floatx(widen_seq(primary))
                lengths.append(wide_seq.shape[0])
                primary_list.append(pad_tensor(wide_seq, max_seq_length))
                tertiary = calc_pairwise_distances(tertiary)
                tertiary_list.append(pad_tensor(tertiary, max_seq_length, two_dim=True))
                mask_list.append(pad_tensor(ter_mask, max_seq_length, two_dim=True))

    batch_primary = primary_2d_to_3d(primary_list)
    batch_tertiary = primary_2d_to_3d(tertiary_list)
    batch_mask = mask_2d_to_3d(mask_list)

    np.save('P:/casp7/casp7/seq_under_' + str(max_seq_length), batch_primary.numpy())
    np.save('P:/casp7/casp7/tertiary_under_' + str(max_seq_length), batch_tertiary.numpy())
    np.save('P:/casp7/casp7/mask_under_' + str(max_seq_length), batch_mask.numpy())
    np.save('P:/casp7/casp7/lengths_under_' + str(max_seq_length), np.asarray(lengths))

    return batch_primary, batch_mask, batch_tertiary, lengths


if __name__ == '__main__':
    #tfrecords = []
    #for i in range(1, 136):
    #    tfrecords.append('P:/casp7/casp7/training/100/' + str(i))
    #primary, mask, tertiary, seq_lengths = gather_data_seq_under_limit(tfrecords)
    primary = np.load('P:/casp7/casp7/seq_under_64.npy')
    tertiary = np.load('P:/casp7/casp7/tertiary_under_64.npy') / 100
    mask = np.load('P:/casp7/casp7/mask_under_64.npy')
    original_lengths = np.load('P:/casp7/casp7/lengths_under_64.npy')
    plt.title('Ground Truth')
    plt.imshow(tertiary[4], cmap='viridis_r', interpolation='nearest')
    plt.colorbar()
    plt.show()