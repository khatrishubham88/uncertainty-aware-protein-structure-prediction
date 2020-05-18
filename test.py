import tensorflow as tf
import numpy as np
import time
from readData_from_TFRec import widen_seq, parse_dataset, NUM_AAS, NUM_EVO_ENTRIES, NUM_DIMENSIONS
from tqdm import tqdm

def widen_seq_unoptimized(seq):
    key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    tensor = []
    for i in range(len(seq)):
        d2 = []
        for j in range(len(seq)):
            # calculating on-hot for one amino acid
            d1 = [1 if (key[x] == seq[i] and key[x] == seq[j]) else 0 for x in range(NUM_AAS)]
            d2.append(d1)
        tensor.append(d2)

    # print(np.array(tensor))
    # print(np.array(tensor).shape)
    return np.array(tensor)  # (LxLx20)

def TestWidenSequence(primarySequence, maxTestShape=100):
    # doing this to make test faster
    if primarySequence.shape[0] < maxTestShape:
        print("Running for shape = %d" % (primarySequence.shape[0]))
        start = time.time()
        wide_seq = widen_seq(primarySequence)
        total = time.time() - start
        proto_seq = tf.make_tensor_proto(wide_seq)
        numpy_seq = tf.make_ndarray(proto_seq)
        print("Done new version in %f sec" % (total))
        start = time.time()
        old_seq = widen_seq_unoptimized(primarySequence)
        total = time.time() - start
        print("Done old version in %f sec" % (total))
        # important otherwise it can cause the difference due to machine precision
        numpy_seq.astype(old_seq.dtype)
        # compare if two array return same value
        print("is same = " + str((numpy_seq == old_seq).all()))
    else:
        print("Primary Sequence shape larger than the maxTestShape \nGiven Sequence shape = %d, maxTestShape = %d"%(primarySequence.shape[0], maxTestShape))


def RunTests(path):
    maxTestShape = 90
    for primary, evolutionary, tertiary, ter_mask in tqdm(parse_dataset(path)):
        TestWidenSequence(primary, maxTestShape)

if __name__ == '__main__':
    # add your test flag here and put it below
    tfrecords_path = '../proteinnet/data/casp7/training/100/1'
    # test function for the optimized function
    RunTests(tfrecords_path)