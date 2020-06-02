import numpy as np
#import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import expand_dim, calc_pairwise_distances, load_npy_binary, output_to_distancemaps
from utils import masked_categorical_cross_entropy, create_crop, random_index
from readData_from_TFRec import parse_tfexample
import glob
import math

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path: list, crop_size=64, datasize=None, features="primary",
                 padding_value=-1, minimum_bin_val=2,
                 maximum_bin_val=22, num_bins=64,
                 batch_size=100, shuffle=False,
                 shuffle_buffer_size=None, random_crop=False, take=None):
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
        self.datasize = None
        if datasize is not None:
            self.datasize = datasize
        else:
            self.datasize = self.fetch_datasize()
        self.shuffle_buffer_size = None
        if shuffle_buffer_size is not None:
            self.shuffle_buffer_size = shuffle_buffer_size
        else:
            self.shuffle_buffer_size = self.batch_size

        if self.shuffle:
            self.raw_dataset = self.raw_dataset.shuffle(self.shuffle_buffer_size)

        self.datafeeder = None
        self.construct_feeder()
        self.iterator = None
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
        self.datafeeder = tf.data.Dataset.from_generator(self.transformation_generator,
                                                         output_types=(tf.float32, tf.float32, tf.float32),
                                                         output_shapes= ((None, None, None, ),
                                                                         (None, None, None, ),
                                                                         (None, None, )))
        self.datafeeder = self.datafeeder.batch(self.batch_size)
        if self.take is not None:
            self.datafeeder = self.datafeeder.take(self.take)

    def fetch_datasize(self):
        count = 0
        for _ in self.raw_dataset:
            count +=1
        return count

    def transformation_generator(self):
        for data in self.raw_dataset:
            primary, evolutionary, tertiary, ter_mask = parse_tfexample(data)
            transformed_batch = DataGenerator.generator_transform(primary, evolutionary, tertiary, ter_mask,
                                                    crop_size=self.crop_size,
                                                    padding_value=self.padding_value,
                                                    minimum_bin_val=self.minimum_bin_val,
                                                    maximum_bin_val=self.maximum_bin_val,
                                                    num_bins=self.num_bins)
            for sample in transformed_batch:
                yield sample # has values (primary, tertiary, tertiary mask)


    @staticmethod
    def generator_transform(primary, evolutionary, tertiary, tertiary_mask, crop_size, padding_value=-1, minimum_bin_val=2,
                        maximum_bin_val=22, num_bins=64):

        # correcting the datatype to avoid errors
        padding_value = float(padding_value)
        minimum_bin_val = float(minimum_bin_val)
        maximum_bin_val = float(maximum_bin_val)
        num_bins = int(num_bins)

        index = random_index(primary, crop_size)
        dist_map = calc_pairwise_distances(tertiary)
        padding_size = math.ceil(primary.shape[0]/crop_size)*crop_size - primary.shape[0]
        # perform cropping + necessary padding
        random_crop = create_crop(primary, dist_map, tertiary_mask, index, crop_size, padding_value, padding_size,
                                    minimum_bin_val, maximum_bin_val, num_bins)
        return random_crop

if __name__=="__main__":
    path = glob.glob("/home/ghalia/Documents/LabCourse/casp7/training/100/1")
    params = {
    "crop_size":64, # this is the LxL
    "datasize":None,
    "features":"primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value":-1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":100,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":True         # if cropping should be random, this has to be implemented later
    }
    dataprovider = DataGenerator(path, **params)
    num_data_points = 0
    for count in dataprovider:
        # print(len(count))
        num_data_points += 1
    print(num_data_points)
