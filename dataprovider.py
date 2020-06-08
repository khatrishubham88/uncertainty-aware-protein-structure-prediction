import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import expand_dim, calc_pairwise_distances, load_npy_binary, output_to_distancemaps
from utils import masked_categorical_cross_entropy, random_index
from readData_from_TFRec import parse_tfexample, create_crop2, create_crop
import glob
import math
from tensorflow.python.ops import array_ops

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, path: list, crop_size=64, 
                 datasize=None, features="primary",
                 padding_value=-1, minimum_bin_val=2,
                 maximum_bin_val=22, num_bins=64,
                 batch_size=100, shuffle=False,
                 shuffle_buffer_size=None, 
                 random_crop=True, take=None, 
                 flattening=True, epochs=1, prefetch = False):
        'Initialization'
        self.path = path
        self.raw_dataset = tf.data.TFRecordDataset(self.path)
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.features = features
        self.padding_value = padding_value
        self.minimum_bin_val = minimum_bin_val
        self.maximum_bin_val = maximum_bin_val
        self.shuffle = shuffle
        self.num_bins = num_bins
        self.random_crop = random_crop
        self.take = take
        self.flattening = flattening
        self.datasize = None
        self.epochs = epochs
        self.prefetch = prefetch
        if datasize is not None:
            self.datasize = datasize
        else:
            self.datasize = self.fetch_datasize()
        self.shuffle_buffer_size = None
        if shuffle_buffer_size is not None:
            self.shuffle_buffer_size = shuffle_buffer_size
        else:
            self.shuffle_buffer_size = self.datasize

        if self.shuffle:
            self.raw_dataset = self.raw_dataset.shuffle(self.shuffle_buffer_size)
        self.datafeeder = None
        self.construct_feeder()
        self.iterator = None
        self.idx_track = []
        self.__iter__()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.datasize / self.batch_size))

    def __getitem__(self, index):
        """
        Needs correction
        """
        raise NotImplementedError('No implemented yet.')

    def __iter__(self):
        self.construct_feeder()
        self.iterator = self.datafeeder.__iter__()
        return self

    def __next__(self):
        output = next(self.iterator)
        return output


    def construct_feeder(self):
        def flat_map(primary, secondary, mask):
            # tf.print(array_ops.shape(mask))
            return (primary, secondary, tf.reshape(mask, shape=(-1,)))

        self.datafeeder = tf.data.Dataset.from_generator(self.transformation_generator,
                                                         output_types=(tf.float32, tf.float32, tf.float32),
                                                        #  output_shapes= ((None, None, None, ),
                                                        #                  (None, None, None, ),
                                                        #                  (None, None, ))
                                                         )
        # batch before map, for vectorization.
        self.datafeeder = self.datafeeder.batch(self.batch_size, drop_remainder=True)
        if self.flattening:
            # parallelizing the flattening
            self.datafeeder = self.datafeeder.map(lambda x, y, z: flat_map(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.take is not None:
            self.datafeeder = self.datafeeder.take(self.take)
        self.datafeeder = self.datafeeder.repeat(self.epochs)
        # apply prefetch for memory optimization
        if self.prefetch:
            self.datafeeder = self.datafeeder.prefetch(tf.data.experimental.AUTOTUNE)

    def fetch_datasize(self):
        count = 0
        for _ in self.raw_dataset:
            count +=1
        return count

    def transformation_generator(self):
        for data in self.raw_dataset:
            primary, evolutionary, tertiary, ter_mask = parse_tfexample(data)
            # print(primary.shape)
            transformed_batch, idx = DataGenerator.generator_transform(primary, evolutionary, tertiary, ter_mask,
                                                    crop_size=self.crop_size,
                                                    padding_value=self.padding_value,
                                                    minimum_bin_val=self.minimum_bin_val,
                                                    maximum_bin_val=self.maximum_bin_val,
                                                    num_bins=self.num_bins,
                                                    random_crop=self.random_crop)
            self.idx_track.append(idx)
            # for sample in transformed_batch:
            yield transformed_batch # has values (primary, tertiary, tertiary mask)


    @staticmethod
    def generator_transform(primary, evolutionary, tertiary, tertiary_mask, crop_size, random_crop=True,
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
        return random_crop, index


# Used for minor testing of data provider
if __name__=="__main__":
    path = glob.glob("../proteinnet/data/casp7/training/100/*")
    params = {
    "crop_size":64, # this is the LxL
    "datasize":None,
    "features":"primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value":-1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":2,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":False,         # if cropping should be random, this has to be implemented later
    "flattening":True,
    "take": 6,
    "epochs":2,
    "prefetch": False
    }
    dataprovider = DataGenerator(path, **params)
    num_data_points = 0
    print(len(dataprovider))
    tensors = []
    for n, count in enumerate(dataprovider):
        # print(len(count))
        # print(count[0].shape)
        # print(count[1].shape)
        # print(count[2].shape)
        if n>=params["take"]:
            # print("printing for element n = {}, and tensor = {}".format(n%params["take"], tf.math.equal(tensors[n%params["take"]], count[0])))
            mismatch = True
            num_mismatch = 0
            for i in range(count[0].shape[0]):
                for j in range(count[0].shape[1]):
                    for k in range(count[0].shape[2]):
                        for l in range(count[0].shape[3]):
                            # print(tf.math.equal(tensors[i-params["take"]], count[0]).numpy()[i,j,k,l] is False)
                            if tf.math.equal(tensors[n%params["take"]], count[0]).numpy()[i,j,k,l]:
                                pass
                            elif not tf.math.equal(tensors[n%params["take"]], count[0]).numpy()[i,j,k,l]:
                                num_mismatch +=1 
                                if mismatch:
                                    print("old val = {}, new val = {}, at i = {}, j = {}, k = {}, l = {}".format(tensors[n%params["take"]][i,j,k,l], count[0][i,j,k,l], i, j, k, l))
                                    mismatch = False
            print("Total mismatch for element n = {} => {}".format(n%params["take"], num_mismatch))
        else:
            tensors.append(count[0])
        # print(count[2].shape)
        # sh = count[1].shape[0:-1]
        # print(sh)
        # reshaped_test = tf.reshape(count[2], shape=sh)
        # print(reshaped_test)
        num_data_points += 1
    print(num_data_points)
