import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import itertools
from utils import calc_pairwise_distances
from utils import to_distogram


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
    # add your test flag here and put it below
    tfrecords_path = '/home/ghalia/Documents/LabCourse/casp7/training/100/1'
    # test function for the optimized function
    for primary, evolutionary, tertiary, ter_mask in parse_dataset(tfrecords_path):
        pass
