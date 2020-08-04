import itertools
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import math
from utils import to_distogram, create_protein_batches
from utils import pad_feature2, calc_pairwise_distances, output_to_distancemaps, contact_map_from_distancemap, accuracy_metric, precision_metric

NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_EVO_ENTRIES = 21


def masking_matrix(input_mask):
    """Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.
      Args:
        input_mask: 0/1 vector indicating whether a position should be masked (0) or not (1)
      Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [SEQ_LENGTH, SEQ_LENGTH]
    """
    mask = tf.convert_to_tensor(input_mask, name='mask')

    #print(tf.size(mask)) #--> Tensor("Size_2:0", shape=(), dtype=int32)
    mask = tf.expand_dims(mask, axis= 0)    # this operation inserts a dimension of size 1 at the dimension index axis
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


def parse_val_tfexample(serialized_input, min_thinning):
    context, features = tf.io.parse_single_sequence_example(serialized_input,
                            context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                            sequence_features={
                                    'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.io.FixedLenSequenceFeature((NUM_EVO_ENTRIES,), tf.float32, allow_missing=True),
                                    'secondary':    tf.io.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.io.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})

    id_ = context['id'][0]
    if int(id_.numpy().decode("utf-8")[0:2]) >= min_thinning:
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
    else:
        return None, None, None, None


def parse_test_tfexample(serialized_input, category):
    context, features = tf.io.parse_single_sequence_example(serialized_input,
                            context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                            sequence_features={
                                    'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.io.FixedLenSequenceFeature((NUM_EVO_ENTRIES,), tf.float32, allow_missing=True),
                                    'secondary':    tf.io.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                                    'tertiary':     tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,),  tf.float32, allow_missing=True),
                                    'mask':         tf.io.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)})

    id_ = context['id'][0]
    if category == 2: #FM
        if id_.numpy().decode("utf-8")[0:2] == 'FM':
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
        else:
            return None, None, None, None
    elif category == 1: #TBM
        if id_.numpy().decode("utf-8")[0:4] == 'TBM' + '#':
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
        else:
            return None, None, None, None
    elif category == 3: #TBM-hard
        if id_.numpy().decode("utf-8")[0:4] == 'TBM-hard'[0:4]:
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
        else:
            return None, None, None, None
    elif category == 4: #TBM/TBM-hard
        if (id_.numpy().decode("utf-8")[0:4] == 'TBM-hard'[0:4] or id_.numpy().decode("utf-8")[0:4] == 'TBM' + '#'):
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
        else:
            return None, None, None, None
    elif category == 5: #all
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
    """This function iterates over all input training files and extracts record information from each single file.
    Use Yield for optimization purpose causes reading when needed.
      Args:
        file_paths: List containing all the paths to training files.
    """
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    for data in raw_dataset:
        parsed_data = parse_tfexample(data)
        yield parsed_data


def parse_val_dataset(file_paths, min_thinning):
    """This function iterates over all input validation files and extracts record information from each single file.
    Use Yield for optimization purpose causes reading when needed.
      Args:
        file_paths: List containing all the paths to validation files.
    """
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    for data in raw_dataset:
        parsed_data = parse_val_tfexample(data, min_thinning)
        yield parsed_data


def parse_test_dataset(file_paths, category):
    """This function iterates over all input test files and extracts record information from each single file.
    Use Yield for optimization purpose causes reading when needed.
      Args:
        file_paths: List containing all the paths to test files.
        category:   Specifies which category (FM, TBM or TBM-hard) we want to use for testing.
    """
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    for data in raw_dataset:
        parsed_data = parse_test_tfexample(data, category)
        yield parsed_data


def widen_seq(seq):
    """Converts a seq into a one-hot tensor (not LxN but LxLxN).
    _aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7',
    'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
      Args:
        seq: Protein sequence.
      Returns:
        Tensor containing widened one-hot encoded sequence information of a protein.
    """
    L = seq.shape[0]
    key = np.arange(start=0, stop=NUM_AAS, step=1)
    wide_tensor = np.zeros(shape=(L, L, NUM_AAS))
    proto_seq = tf.make_tensor_proto(seq)
    numpy_seq = tf.make_ndarray(proto_seq)
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(key.reshape(-1, 1))
    encoding = enc.transform(key.reshape(-1, 1)).toarray()
    for i in range(NUM_AAS):
        pos = np.argwhere(numpy_seq == i)
        for j, k in itertools.product(pos, repeat=2):
            wide_tensor[j, k, :] = encoding[i, :]

    return tf.convert_to_tensor(wide_tensor, dtype=tf.float32)


def widen_pssm(pssm):
    """
    Widens the L x L PSSM matrix into LxLxN shape.
      Args:
        pssm: PSSM matrix.
      Returns:
        Tensor containing widened one-hot encoded information of a PSSM matrix.
    """
    L = pssm.shape[0]
    wide_tensor = np.zeros(shape=(L, L, NUM_EVO_ENTRIES))
    proto_pssm = tf.make_tensor_proto(pssm)
    npy_pssm = tf.make_ndarray(proto_pssm)
    for i in range(npy_pssm.shape[0]):
        for j in range(npy_pssm.shape[0]):
            new_feature_vec = np.multiply(npy_pssm[i], npy_pssm[j])/2
            wide_tensor[i, j, :] = new_feature_vec
    return tf.convert_to_tensor(wide_tensor, dtype=tf.float32)


def create_crop2(primary, evolutionary, dist_map, tertiary_mask, features, index, crop_size, padding_value,
                 padding_size, minimum_bin_val, maximum_bin_val, num_bins):
    """If the sequence length is bigger than the crop_size this function crops a random (crop_size x crop_size) window
    from the calculated features. Otherwise, it pads the features to the crop_size and returns them.
      Args:
        primary:            Protein sequence.
        evolutionary:       Evolutionary information of the protein.
        dist_map:           Inter-residual distances in Angstrom.
        tertiary_mask:      Mask matrix.
        features:           If feature is equal to 'primary', we do not use evolutionary information to construct
                            the input to the network.
        index:              List containing two random indices used for cropping.
        crop_size:          Integer determining how many residuals are used in each crop.
        padding_value:      Integer that is used for padding.
        padding_size:       Integer determining the maximum number of paddings needed along a dimension.
        minimum_bin_val:    Minimum value of inter-residual distances used for discretization in Angstrom.
        maximum_bin_val:    Maximum value of inter-residual distances used for discretization in Angstrom.
        num_bins:           Number of bins used for the discretization of the inter-residual distances.
      Returns:
        Tuple consisting of crops of primary input, ground truth and mask.
    """
    if features == 'primary':
        if primary.shape[0] >= crop_size:
            primary = widen_seq(primary)
            primary_crop = primary[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size, :]
            dist_map_crop = dist_map[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
            distogram_crop = to_distogram(dist_map_crop, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
            tertiary_mask_crop = tertiary_mask[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
            return (primary_crop, distogram_crop, tertiary_mask_crop)
        else:
            primary = widen_seq(primary)
            primary = pad_feature2(primary, crop_size, padding_value, padding_size, 2)
            dist_map = pad_feature2(dist_map, crop_size, padding_value, padding_size, 2)
            distogram = to_distogram(dist_map, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
            tertiary_mask = pad_feature2(tertiary_mask, crop_size, 0, padding_size, 2)
            return (primary, distogram, tertiary_mask)
    else:
        if features == 'pri-evo':
            if primary.shape[0] >= crop_size:
                primary = widen_seq(primary)
                evol = widen_pssm(evolutionary)
                primary_crop = primary[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size, :]
                evol_crop = evol[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size, :]
                pri_evol_crop = tf.concat([primary_crop, evol_crop],axis=2)
                dist_map_crop = dist_map[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
                distogram_crop = to_distogram(dist_map_crop, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
                tertiary_mask_crop = tertiary_mask[index[0]:index[0]+crop_size, index[1]:index[1]+crop_size]
                return (pri_evol_crop, distogram_crop, tertiary_mask_crop)
            else:
                primary = widen_seq(primary)
                evol = widen_pssm(evolutionary)
                primary = pad_feature2(primary, crop_size, padding_value, padding_size, 2)
                evol = pad_feature2(evol, crop_size, padding_value, padding_size, 2)
                pri_evol = tf.concat([primary, evol],axis=2)
                dist_map = pad_feature2(dist_map, crop_size, padding_value, padding_size, 2)
                distogram = to_distogram(dist_map, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
                tertiary_mask = pad_feature2(tertiary_mask, crop_size, 0, padding_size, 2)
                return (pri_evol, distogram, tertiary_mask)

def prepare_test_set(path:list, modelling_group:int, crop_size, padding_value, min_bin_val, max_bin_val, num_bins, batch_size):
    X = []
    y = []
    mask = []
    for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(path, modelling_group):
        if (primary != None):
            primary_2D = widen_seq(primary)
            pssm = widen_pssm(evolutionary)
            dist_map = calc_pairwise_distances(tertiary)
            padding_size = math.ceil(primary.shape[0]/crop_size)*crop_size - primary.shape[0]
            padded_primary = pad_feature2(primary_2D, crop_size, padding_value, padding_size, 2)
            padded_evol = pad_feature2(pssm, crop_size, padding_value, padding_size, 2)
            padded_dist_map = pad_feature2(dist_map, crop_size, padding_value, padding_size, 2)
            padded_mask = pad_feature2(ter_mask, crop_size, padding_value, padding_size, 2)
            batches = create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask,
                                           crop_size, crop_size, min_bin_val,
                                           max_bin_val, num_bins)
            for i, batch in enumerate(batches):
                X.append(batch[0])            # batch[0] of type eager tensor
                y.append(batch[1])            # batch[1] of type ndarray
                mask.append(batch[2])         # batch[2] of type eager tensor
    X = tf.convert_to_tensor(X)
    y = np.asarray(y)
    mask = tf.convert_to_tensor(mask)
    mask = np.asarray(mask)
    
    if X.shape[0] % batch_size != 0:
        drop_samples = X.shape[0] - ((X.shape[0] // batch_size) * batch_size)
        X = X[0:X.shape[0] - drop_samples, :, :]
        mask = mask[0:mask.shape[0] - drop_samples, :, :]
        y = y[0:y.shape[0] - drop_samples, :, :, :]
    return X, y, mask