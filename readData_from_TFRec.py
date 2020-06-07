import math
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from utils import random_index, calc_pairwise_distances, create_crop


NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_EVO_ENTRIES = 21

N = 20
key = np.arange(start=0, stop=N, step=1)
enc = OneHotEncoder(handle_unknown='error')
enc.fit(key.reshape(-1, 1))


def masking_matrix(input_mask):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.
    Args:
        input_mask: 0/1 vector indicating whether a position should be masked (0) or not (1)
    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [SEQ_LENGTH, SEQ_LENGTH]
    """

    mask = tf.convert_to_tensor(input_mask, name='mask')

    #print(tf.size(mask)) #--> Tensor("Size_2:0", shape=(), dtype=int32)
    mask = tf.expand_dims(mask, axis= 0) #--> this operation inserts a dimension of size 1 at the dimension index axis
    #print(tf.size(mask)) #--> Tensor("Size_3:0", shape=(), dtype=int32)
    base = tf.ones([tf.size(mask), tf.size(mask)])

    matrix_mask = base * mask * tf.transpose(mask)

    return matrix_mask


def parse_tfexample(serialized_input):
    context, features = tf.io.parse_single_sequence_example(serialized_input,
                            context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                            sequence_features={
                                    'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.io.FixedLenSequenceFeature((NUM_EVO_ENTRIES,), tf.float32, allow_missing=True),
                                    'secondary':    tf.io.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.io.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})

    id_ = context['id'][0]
    primary = tf.dtypes.cast(features['primary'][:, 0], tf.int32)
    evolutionary = features['evolutionary']
    secondary = tf.dtypes.cast(features['secondary'][:, 0], tf.int32)
    tertiary = features['tertiary']
    mask = features['mask'][:, 0]

    pri_length = tf.size(primary)
    # Generate tertiary masking matrix--if mask is missing then assume all residues are present
    mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length]))
    ter_mask = masking_matrix(mask)
    return primary, evolutionary, tertiary, ter_mask


def parse_dataset(file_paths):
    """
    This function iterates over all input files
    and extract record information from each single file
    Use Yield for optimization purpose causes reading when needed
    """

    raw_dataset = tf.data.TFRecordDataset(file_paths)
    for data in raw_dataset:
        parsed_data = parse_tfexample(data)
        yield parsed_data

"""
def widen_pssm(pssm, seq):
    "" Converts a seq into a tensor. Not LxN but LxLxN.
        Multiply cell i,j x j,i to create an LxLxN tensor.
    ""
    key = "HRKDENQSYTCPAVLIGFWM"
    key_alpha = "ACDEFGHIKLMNPQRSTVWY"
    tensor = []
    for i in range(self.pad_size):
        d2 = []
        for j in range(self.pad_size):
            # Check first for lengths (dont want index_out_of_range)
            if j<len(seq) and i<len(seq):
                d1 = [aa[i]*aa[j] for aa in pssm]
            else:
                d1 = [0 for i in range(n)]

            # Append pssm[i]*pssm[j]
            if j<len(seq) and i<len(seq):
                d1.append(pssm[key_alpha.index(seq[i])][i] *
                          pssm[key_alpha.index(seq[j])][j])
            else:
                d1.append(0)
            # Append manhattan distance to diagonal but reversed (center=0, xtremes=1)
            d1.append(1 - abs(i-j)/self.crop_size)

            d2.append(d1)
        tensor.append(d2)

    return np.array(tensor)
"""


def parse_val_test_set(file_paths):
    """
    This function iterates over all input files
    and extract record information from each single file
    Use Yield for optimization purpose causes reading when needed
    """

    raw_dataset = tf.data.TFRecordDataset(file_paths)
    for data in raw_dataset:
        parsed_data = parse_val_test_tfexample(data)
        yield parsed_data


def parse_val_test_tfexample(serialized_input):
    context, features = tf.io.parse_single_sequence_example(serialized_input,
                            context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                            sequence_features={
                                    'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.io.FixedLenSequenceFeature((NUM_EVO_ENTRIES,), tf.float32, allow_missing=True),
                                    'secondary':    tf.io.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.io.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})

    id_ = context['id'][0]
    primary = tf.dtypes.cast(features['primary'][:, 0], tf.int32)
    evolutionary = features['evolutionary']
    secondary = tf.dtypes.cast(features['secondary'][:, 0], tf.int32)
    tertiary = features['tertiary']
    mask = features['mask'][:, 0]

    pri_length = tf.size(primary)
    # Generate tertiary masking matrix--if mask is missing then assume all residues are present
    mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length]))
    ter_mask = masking_matrix(mask)
    return id_, primary, evolutionary, tertiary, ter_mask


def data_transformation(primary, evolutionary, tertiary, tertiary_mask, crop_size, random_crop=True,
                        padding_value=-1, minimum_bin_val=2, maximum_bin_val=22, num_bins=64):

    # correcting the datatype to avoid errors
    padding_value = float(padding_value)
    minimum_bin_val = float(minimum_bin_val)
    maximum_bin_val = float(maximum_bin_val)
    num_bins = int(num_bins)

    if random_crop:
        index = random_index(primary, crop_size)
    else:
        index = [0, 0]
    dist_map = calc_pairwise_distances(tertiary)
    padding_size = math.ceil(primary.shape[0]/crop_size)*crop_size - primary.shape[0]
    # perform cropping + necessary padding
    random_crop = create_crop(primary, dist_map, tertiary_mask, index, crop_size, padding_value, padding_size,
                              minimum_bin_val, maximum_bin_val, num_bins)

    return random_crop


def generate_val_test_numpy_binaries(source_val_path, crop_size, random_crop, crops_per_sample, padding_value,
                                     minimum_bin_val, maximum_bin_val, num_bins):
    primary_list = []
    tertiary_list = []
    ter_mask_list = []
    for id_, primary, evolutionary, tertiary, ter_mask in tqdm(parse_val_test_set(source_val_path)):
        sample_thinning = int(id_.numpy().decode("utf-8")[0:2])
        if sample_thinning >= 50:
            for _ in range(crops_per_sample):
                primary_crop, distogram_crop, mask_crop = data_transformation(primary, evolutionary, tertiary,
                                                                              ter_mask, crop_size, random_crop,
                                                                              padding_value, minimum_bin_val,
                                                                              maximum_bin_val, num_bins)
                primary_list.append(primary_crop)
                tertiary_list.append(distogram_crop)
                ter_mask_list.append(mask_crop)

    return primary_list, tertiary_list, ter_mask_list


if __name__ == '__main__':
    path = 'P:/casp7/casp7/validation/*'
    primary_list = []
    tertiary_list = []
    ter_mask_list = []
    for primary, evolutionary, tertiary, ter_mask in tqdm(parse_dataset(path)):
        print(primary)
        break
    #primary, tertiary, mask = generate_val_test_numpy_binaries(path, 64, True, 100, -1, 2, 22, 64)