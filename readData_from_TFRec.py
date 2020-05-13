import os
import tensorflow as tf
import numpy as np

NUM_AAS = 20
NUM_DIMENSIONS = 3


def parse_one_tfrecord(serialized_input, num_evo_entries=21):
    context, features = tf.io.parse_single_sequence_example(serialized_input,
                            context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
                            sequence_features={
                                    'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                                    'evolutionary': tf.io.FixedLenSequenceFeature((num_evo_entries,), tf.float32, allow_missing=True),
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

    return id_, primary, evolutionary, secondary, tertiary, pri_length


def parse_dataset(file_paths, num_evo_entries=21):
    """ This function iterates over all input files
    and extract record information from each single file"""
    tfrecord_dataset = tf.data.TFRecordDataset(file_paths)
    for raw_example in iter(tfrecord_dataset):
        id_, primary, evolutionary, secondary, tertiary, pri_length = parse_one_tfrecord(raw_example, num_evo_entries=21)
        print(tertiary)
        widen_seq(primary)
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
            d1 = [1 if (j<len(seq) and i<len(seq) and key[x] == seq[i] and key[x] == seq[j])
                else 0 for x in range(NUM_AAS)]

            d2.append(d1)
        tensor.append(d2)

    #print(np.array(tensor))
    #print(np.array(tensor).shape)
    return np.array(tensor)

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
