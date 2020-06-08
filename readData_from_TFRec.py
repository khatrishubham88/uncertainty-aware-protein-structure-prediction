import glob
import itertools
import math
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from utils import calc_pairwise_distances, random_index, to_distogram
from utils import pad_feature, pad_feature2


NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_EVO_ENTRIES = 21


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

def widen_seq(seq):
    """
    _aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7',
     'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
    """
    """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
    L = seq.shape[0]
    N = 20
    key = np.arange(start=0,stop=N,step=1)
    wide_tensor = np.zeros(shape=(L,L,N))
    proto_seq = tf.make_tensor_proto(seq)
    numpy_seq = tf.make_ndarray(proto_seq)
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(key.reshape(-1,1))
    encoding = enc.transform(key.reshape(-1,1)).toarray()
    for i in range(N):
        pos = np.argwhere(numpy_seq==i)
        for j,k in itertools.product(pos, repeat=2):
            wide_tensor[j,k,:] = encoding[i,:]
    return tf.convert_to_tensor(wide_tensor, dtype=tf.int64)


def create_protein_batches(primary_2D, padded_tertiary, padded_mask, stride):
    batches = []
    for x in range(0,primary_2D.shape[0],stride):
        for y in range(0,primary_2D.shape[0],stride):
            primary_2D_crop = primary_2D[x:x+stride, y:y+stride, :]
            padded_tertiary_crop = padded_tertiary[x:x+stride, y:y+stride]
            # padded_tertiary_crop = to_distogram(padded_tertiary_crop, 2, 22, 64)
            mask_crop = padded_mask[x:x+stride, y:y+stride]
            batches.append((primary_2D_crop, padded_tertiary_crop, mask_crop))

    return batches


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
    
    
def create_crop2(primary, dist_map, tertiary_mask, index, crop_size, padding_value, padding_size, minimum_bin_val,
                    maximum_bin_val, num_bins):
    if primary.shape[0] >= crop_size:
        primary = widen_seq(primary)
        primary = primary[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size,:]
        dist_map = dist_map[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        dist_map = to_distogram(dist_map, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
        tertiary_mask = tertiary_mask[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
        return (primary, dist_map, tertiary_mask)
    else:
        primary = widen_seq(primary)
        primary = pad_feature2(primary, crop_size, padding_value, padding_size, 2)
        dist_map = pad_feature2(dist_map, crop_size, padding_value, padding_size, 2)
        dist_map = to_distogram(dist_map, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
        tertiary_mask = pad_feature2(tertiary_mask, crop_size, 0, padding_size, 2)
        return (primary, dist_map, tertiary_mask)


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


def data_transformations(primary, evolutionary, tertiary, tertiary_mask, crop_size, random_crop=True,
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
    random_crop = create_crop2(primary, dist_map, tertiary_mask, index, crop_size, padding_value, padding_size,
                                minimum_bin_val, maximum_bin_val, num_bins)
    return random_crop


def generate_val_test_npy_binary(path, min_thinning, crop_size, num_crops_per_sample, random_crop, padding_value,
                                 minimum_bin_val, maximum_bin_val, num_bins):
    primary_list = []
    tertiary_list = []
    mask_list = []
    for id_, primary, evolutionary, tertiary, ter_mask in tqdm(parse_val_test_set(path)):
        sample_thinning = id_.numpy().decode("utf-8")[0:2]
        if int(sample_thinning) >= min_thinning:
            for i in range(0, num_crops_per_sample):
                random_crop_tuple = data_transformations(primary=primary, evolutionary=evolutionary, tertiary=tertiary,
                                                         tertiary_mask=ter_mask, crop_size=crop_size,
                                                         random_crop=random_crop, padding_value=padding_value,
                                                         minimum_bin_val=minimum_bin_val,
                                                         maximum_bin_val=maximum_bin_val, num_bins=num_bins)
                primary_list.append(random_crop_tuple[0])
                tertiary_list.append(random_crop_tuple[1])
                mask_list.append(random_crop_tuple[2])

    return primary_list, tertiary_list, mask_list


if __name__ == '__main__':
    path = glob.glob("P:/casp7/casp7/validation/*")
    min_thinning = 50
    crop_size = 64
    num_crops_per_sample = 100
    random_crop = True
    padding_value = 0
    minimum_bin_val = 2
    maximum_bin_val = 22
    num_bins = 64
    primary_list, tertiary_list, mask_list = generate_val_test_npy_binary(path, min_thinning, crop_size,
                                                                          num_crops_per_sample, random_crop,
                                                                          padding_value, minimum_bin_val,
                                                                          maximum_bin_val, num_bins)
    print(len(primary_list))
    print(len(tertiary_list))
    print(len(mask_list))