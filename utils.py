import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from readData_from_TFRec import widen_seq


def load_npy_binary(path):
    return np.load(path)


def masked_categorical_cross_entropy():
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = tf.keras.losses.CategoricalCrossentropy()
        l = loss(y_true, y_pred)

        return l

    return loss


def masked_categorical_cross_entropy_test():
    # mask = K.variable()
    kerasloss = tf.keras.losses.CategoricalCrossentropy()

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        l = kerasloss(y_true, y_pred)
        return l

    return loss


def expand_dim(low_dim_tensor):
    return K.stack(low_dim_tensor, axis=0)


def output_to_distancemaps(output, min_angstrom, max_angstrom, num_bins):
    output = K.eval(output)
    distance_maps = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))

    bins = np.linspace(min_angstrom, max_angstrom, num_bins)
    values = np.argmax(output, axis=3)
    for batch in range(distance_maps.shape[0]):
        distance_maps[batch] = bins[values[batch]]

    return distance_maps


def pad_tensor(tensor, shape):
    if isinstance(shape, int):
        shape = tuple([shape])
    else:
        shape = tuple(shape)
    dim = len(shape)
    padded_tensor = np.zeros(shape)
    if dim == 1:
        padded_tensor[0:tensor.shape[0]] = tensor
    elif dim == 2:
        padded_tensor[0:tensor.shape[0], 0:tensor.shape[0]] = tensor

    return padded_tensor


def pad_primary(tensor, shape):
    #AA space is betwen 0-19 --> paddings have Id 20
    curr_length = tensor.shape[0]
    padded_tensor = np.full(shape, 20)
    padded_tensor[0:curr_length] = tensor

    return padded_tensor


def pad_tertiary(tensor, shape):
    curr_length = tensor.shape[0]
    padded_tensor = np.zeros(shape=shape)
    padded_tensor[0:curr_length, 0:curr_length] = tensor

    return padded_tensor


def pad_mask(tensor, shape):
    curr_length = tensor.shape[0]
    padded_tensor = np.zeros(shape=shape)
    padded_tensor[0:curr_length, 0:curr_length] = tensor

    return padded_tensor

"""Calculates the distance between two AA
using three different approaches:
    1- distance between the N atoms of the two AA
    2- distance between the C-alpha atoms of the two AA
    3- distance between the C-prime atoms of the two AA
"""
def calc_distance(aa1, aa2):
    aa1_N_pos = (aa1[0][0], aa1[0][1], aa1[0][2])
    aa2_N_pos = (aa2[0][0], aa2[0][1], aa2[0][2])

    aa1_C_alpha_pos = (aa1[1][0], aa1[1][1], aa1[1][2])
    aa2_C_alpa_pos = (aa2[1][0], aa2[1][1], aa2[1][2])

    aa1_C_prime_pos = (aa1[2][0], aa1[2][1], aa1[2][2])
    aa2_C_prime_pos = (aa2[2][0], aa2[2][1], aa2[2][2])

    N_distance = math.sqrt(
        (aa2_N_pos[0] - aa1_N_pos[0]) ** 2 + (aa2_N_pos[1] - aa1_N_pos[1]) ** 2 + (aa2_N_pos[2] - aa1_N_pos[2]) ** 2)
    C_alpha_distance = math.sqrt(
        (aa2_C_alpa_pos[0] - aa1_C_alpha_pos[0]) ** 2 + (aa2_C_alpa_pos[1] - aa1_C_alpha_pos[1]) ** 2 + (
                    aa2_C_alpa_pos[2] - aa1_C_alpha_pos[2]) ** 2)
    C_prime_distance = math.sqrt(
        (aa2_C_prime_pos[0] - aa1_C_prime_pos[0]) ** 2 + (aa2_C_prime_pos[1] - aa1_C_prime_pos[1]) ** 2 + (
                    aa2_C_prime_pos[2] - aa1_C_prime_pos[2]) ** 2)

    return N_distance, C_alpha_distance, C_prime_distance


"""This function takes as input the two C-alpha corrdinates
of two AAs and returns the angstrom distance between them
input: coord1 [x, y, z], coord2 [x, y, z]
"""
def calc_calpha_distance(coord1, coord2):
    C_alpha_distance = math.sqrt(
        (coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2 + (coord2[2] - coord1[2]) ** 2)

    return C_alpha_distance


"""Calculates the pairwise distances between
AAs of a protein and returns the distance map in angstrom.
input: tertiary is a tensor of shape (seq_len, 3)
"""
def calc_pairwise_distances(tertiary):
    tertiary_numpy = tertiary.numpy() / 100
    c_alpha_coord = []
    for index, coord in enumerate(tertiary_numpy):
        # Extract only c-alpha coordinates
        if (index % 3 == 1):
            c_alpha_coord.append(coord.tolist())

    # Initialize the distance matrix of shape (len_seq, len_seq)
    distance_matrix = []
    for i, coord1 in enumerate(c_alpha_coord):
        dist = []
        for j, coord2 in enumerate(c_alpha_coord):
            distance = calc_calpha_distance(coord1, coord2)
            dist.append(distance)
        distance_matrix.append(dist)

    return tf.convert_to_tensor(distance_matrix)


"""Returns the distogram tensor LxLxnum_bins from the distance map LxL.
Input: distance_map: LxL distance matrx in angstrom
       min: minimum value for the histogram in angstrom
       max: maximum value for the histogram in angstrom
       num_bins: integer number
"""
def to_distogram(distance_map, min_val, max_val, num_bins):
    assert min_val >= 0.0
    assert max_val > 0.0
    histo_range = max_val-min_val

    distance_map = np.clip(distance_map, a_min=min_val, a_max=max_val)
    distance_map = np.int32(np.floor((num_bins-1)*(distance_map-min_val)/(histo_range)))
    distogram = np.eye(num_bins)[distance_map]

    return distogram


def random_index(primary, crop_size):
    index = []
    if primary.shape[0] <= crop_size:
        index.extend([0, 0])
    else:
        index.extend([np.random.randint(0, primary.shape[0] - crop_size),
                      np.random.randint(0, primary.shape[0] - crop_size)])
    return index


def pad_feature(feature, crop_size, padding_value, padding_size):
    # pad on left and bottom
    padding = tf.constant([[0, padding_size]])
    rank = tf.rank(feature).numpy()
    padding = tf.repeat(padding, rank, axis=0)
    padded_feature = tf.pad(feature, padding, constant_values=tf.cast(padding_value, feature.dtype))
    return padded_feature

def create_crop(primary, dist_map, tertiary_mask, index, crop_size, padding_value, padding_size, minimum_bin_val,
                    maximum_bin_val, num_bins):
    #if primary.shape[0] % crop_size != 0:
    if primary.shape[0] >= crop_size:
        #padded_primary = pad_feature(primary, crop_size, padding_value, padding_size)
        #padded_dist_map = pad_feature(dist_map, crop_size, padding_value, padding_size)
        #padded_ter_mask = pad_feature(tertiary_mask, crop_size, padding_value, padding_size)
        primary_2D = widen_seq(primary)
        # create crops from padded 2D features
        primary_2D_crop = primary_2D[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size,:]
        dist_map_crop = dist_map[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        ter_mask_crop = tertiary_mask[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        distogram_crop = to_distogram(dist_map_crop, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
        return primary_2D_crop, distogram_crop, ter_mask_crop

    else:
        padded_primary = pad_feature(primary, crop_size, padding_value, padding_size)
        padded_dist_map = pad_feature(dist_map, crop_size, padding_value, padding_size)
        padded_ter_mask = pad_feature(tertiary_mask, crop_size, 0, padding_size)
        primary_2D = widen_seq(padded_primary)
        primary_2D_crop = primary_2D[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size,:]
        dist_map_crop = padded_dist_map[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        ter_mask_crop = padded_ter_mask[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        distogram_crop = to_distogram(dist_map_crop, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
        return primary_2D_crop, distogram_crop, ter_mask_crop