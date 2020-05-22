import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def masked_categorical_cross_entropy(y_true, y_pred, mask):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred)
    loss = -K.sum(loss, axis=3) * K.cast_to_floatx(mask)
    loss = K.sum(K.sum(K.sum(loss))) / y_true.size()[0]

    return loss


def mask_2d_to_3d(masks_2d):
    return K.stack(masks_2d, axis=0)


def primary_2d_to_3d(primary_2d):
    return K.stack(primary_2d, axis=0)


def output_to_distancemaps(output, min_angstrom, max_angstrom, num_bins, trash_class=False):
    output = K.eval(output)
    distance_maps = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))

    bins = np.linspace(min_angstrom, max_angstrom, num_bins + 1)
    print(bins)
    values = np.argmax(output, axis=3)
    for batch in range(distance_maps.shape[0]):
        distance_maps[batch] = (bins[values[batch]] + bins[values[batch] + 1]) / 2

    return distance_maps


def pad_tensor(tensor, dim=100, two_dim=False):
    curr_length = tensor.shape[0]
    if not two_dim:
        num_channels = tensor.shape[2]
        padded_tensor = np.zeros(shape=(dim, dim, num_channels))
        padded_tensor[0:curr_length, 0:curr_length, :] = tensor
    elif two_dim:
        padded_tensor = np.zeros(shape=(dim, dim))
        padded_tensor[0:curr_length, 0:curr_length] = tensor

    return tf.convert_to_tensor(padded_tensor)


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
of two AAs and returns the distance between them
input: coord1 [x, y, z], coord2 [x, y, z]
"""


def calc_calpha_distance(coord1, coord2):
    C_alpha_distance = math.sqrt(
        (coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2 + (coord2[2] - coord1[2]) ** 2)

    return C_alpha_distance


"""Calculates the pairwise distances between
AAs of a protein and returns the distance map.
input: tertiary is a tensor of shape (seq_len, 3)
"""


def calc_pairwise_distances(tertiary):
    tertiary_numpy = tertiary.numpy()
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


if __name__ == '__main__':
    out = K.softmax(K.random_normal(shape=(32, 64, 64, 64)), axis=3)
    dist = output_to_distancemaps(out, 2, 22, 64)
    print(np.argmax(out[31][32][32][:]))
    print(dist[31][32][32])