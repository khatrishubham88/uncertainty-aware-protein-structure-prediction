import numpy as np
import tensorflow as tf
import time

import glob

from readData_from_TFRec import widen_seq, parse_dataset, NUM_AAS, NUM_EVO_ENTRIES, NUM_DIMENSIONS
from network import ResNet
from tqdm import tqdm
from dataprovider import DataGenerator
from old_modules import DataGeneratorOld, widen_seq_unoptimized


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

def TestNetwork():
    nn = ResNet(input_channels=64, output_channels=64, num_blocks=[4, 4], num_channels=[64, 32], dilation=[1, 2, 4, 8],
                batch_size=32, dropout_rate=0.15)
    model = nn.model()

def TestDataProvider(path, params):
    dataprovider1 = DataGenerator(path, **params)
    dataprovider2 = DataGeneratorOld(path, **params)
    for data1, data2 in zip(dataprovider1, dataprovider2):
        total_true = data1[2].shape[0]
        sh = data1[1].shape[0:-1]
        reshaped_test = tf.reshape(data1[2], shape=sh)
        counted_true = tf.math.reduce_sum(tf.cast(tf.math.equal(data2[2], reshaped_test), dtype=tf.int32))
        diff = total_true - counted_true.numpy()
        if diff>0:
            raise ValueError("The reshaping introduces error!")



def RunTests(path, test_names):
    params = {
    "dim":(128,128), # this is the LxL
    "datasize":None, 
    "features":"primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value":-1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":1,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":False,         # if cropping should be random, this has to be implemented later
    "flattening" : True,        # if flattten the mask
    } 
    if "datapipeline" in test_names:
        start_time = time.time()
        print("\n\n\nStarting Datapipeline test for flatenning of mask!\n\n\n")
        if not isinstance(path, list):
            path = [path]
        TestDataProvider(path, params)
        print("\n\n\nSuccessfully finished testing of datapipeline in %f sec!\n\n\n"%(time.time() - start_time))
    if "widenseq" in test_names:
        start_time = time.time()
        print("\n\n\nStarting widenseq test for optimized and unoptimized version!\n\n\n")
        maxTestShape = 90
        for primary, evolutionary, tertiary, ter_mask in tqdm(parse_dataset(path)):
            TestWidenSequence(primary, maxTestShape)
        print("\n\n\nSuccessfully finished widen_seq testing in %f sec\n\n\n"%(time.time() - start_time))
if __name__ == '__main__':
    # add your test flag here and put it below
    tfrecords_path = '../proteinnet/data/casp7/training/100/1'
    # test function for the optimized function
    test_names = [
        "datapipeline", 
        "widenseq"
        ]
    RunTests(tfrecords_path, test_names)
