import os
import tensorflow as tf
import numpy as np
from utils import calc_pairwise_distances

NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_EVO_ENTRIES = 21


def masking_matrix(input_mask):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.
    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)
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
    primary =   tf.dtypes.cast(features['primary'][:, 0], tf.int32)
    evolutionary =          features['evolutionary']
    secondary = tf.dtypes.cast(features['secondary'][:, 0], tf.int32)
    tertiary =              features['tertiary']
    mask =                  features['mask'][:, 0]

    pri_length = tf.size(primary)
    # Generate tertiary masking matrix--if mask is missing then assume all residues are present
    mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones([pri_length]))
    ter_mask = masking_matrix(mask)
    return primary, evolutionary, tertiary, ter_mask



def parse_dataset(file_paths):
    """ This function iterates over all input files
    and extract record information from each single file"""
    raw_dataset = tf.data.TFRecordDataset(file_paths)
    raw_dataset = raw_dataset.map(lambda raw: parse_tfexample(raw)) #each item in raw_dataset is a tuple of tensors
    #raw_dataset.create_batch(batch_size, crop_size) --> at this call the cropping should be done
    for data in raw_dataset:
        tertiary = data[2]
        #print(data[0])
        #res = widen_seq(data[0])
        #print(res.shape)
        print(calc_pairwise_distances(tertiary))

        #aa1_backpone_pos = np.array((tertiary[0].tolist(), tertiary[1].tolist(), tertiary[2].tolist()))
        #aa2_backpone_pos = np.array((tertiary[3].tolist(), tertiary[4].tolist(), tertiary[5].tolist()))
        #print(aa1_backpone_pos)
        #print(aa2_backpone_pos)
        #N_distane, C_alpha_distance, C_prime_distance = calc_distance(aa1_backpone_pos, aa2_backpone_pos)
        #print('N_distance:', N_distane, 'C_alpha_distance :', C_alpha_distance, 'C_prime_distance :', C_prime_distance)
        break


def widen_seq(seq):
    """
    _aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7',
     'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
    """
    """ Converts a seq into a one-hot tensor. Not LxN but LxLxN"""
    key = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    tensor = []
    for i in range(len(seq)):
        d2 = []
        for j in range(len(seq)):
            # calculating on-hot for one amino acid
            d1 = [1 if (key[x] == seq[i] and key[x] == seq[j])
                else 0 for x in range(NUM_AAS)]

            d2.append(d1)
        tensor.append(d2)

    #print(np.array(tensor))
    #print(np.array(tensor).shape)
    return np.array(tensor) #(LxLx20)

def widen_pssm(pssm, seq):
    """ Converts a seq into a tensor. Not LxN but LxLxN.
        Multiply cell i,j x j,i to create an LxLxN tensor.
    """
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

if __name__ == '__main__':

    tfrecords_path = '/home/ghalia/Documents/LabCourse/casp7/training/100/2'
    parse_dataset(tfrecords_path)
