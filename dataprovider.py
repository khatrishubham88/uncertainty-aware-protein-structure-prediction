import numpy as np
#import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import expand_dim, calc_pairwise_distances, to_distogram, load_npy_binary, output_to_distancemaps
from utils import pad_mask, pad_primary, pad_tertiary, masked_categorical_cross_entropy
from readData_from_TFRec import widen_seq, parse_tfexample, create_protein_batches
import glob
import math

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path: list, dim=(64,64), datasize=None, features="primary", 
                 padding_value=-1, minimum_bin_val=2, 
                 maximum_bin_val=22, num_bins=64, 
                 batch_size=100, shuffle=False, 
                 shuffle_buffer_size=None, random_crop=False, take=None):
        'Initialization'
        self.path = path
        self.raw_dataset = tf.data.TFRecordDataset(self.path)
        self.dim = dim
        self.stride = dim[0]
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
        return next(self)
    
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
                                                    stride=self.stride, 
                                                    padding_value=self.padding_value, 
                                                    minimum_bin_val=self.minimum_bin_val, 
                                                    maximum_bin_val=self.maximum_bin_val, 
                                                    num_bins=self.num_bins)
            for subset in transformed_batch:
                yield subset # has values (primary, tertiary, tertiary mask)
    
    @staticmethod
    def generator_transform(primary, evolutionary, tertiary, tertiary_mask, stride, padding_value=-1, minimum_bin_val=2, 
                        maximum_bin_val=22, num_bins=64):
    
        # correcting the datatype to avoid errors
        stride = int(stride)
        padding_value = float(padding_value)
        minimum_bin_val = float(minimum_bin_val)
        maximum_bin_val = float(maximum_bin_val)
        num_bins = int(num_bins)
        
        # detecting the number of crops
        crops_per_seq = primary.shape[0] // stride
        if (primary.shape[0] % stride > 0) and (crops_per_seq > 0):
            crops_per_seq += 1
        total_crops = crops_per_seq * crops_per_seq
        
        # compute padding necessary for this protein
        # Find the number of padding elements
        num_padding = math.ceil(primary.shape[0]/stride)*stride - primary.shape[0] 
        # pad on left and bottom
        padding = tf.constant([[0, num_padding]])
        
        # primary transformation
        # compute the rank of tensor to apply padding
        primary_rank = tf.rank(primary).numpy()
        primary_padding = tf.repeat(padding, primary_rank, axis=0)
        primary = tf.pad(primary, primary_padding, constant_values=tf.cast(padding_value, primary.dtype))
        # widen primary sequence to convert it to 2D
        primary = widen_seq(primary)
        # cast it to float for the model
        primary = K.cast_to_floatx(primary)
        
        # tertiary trnsformation
        tertiary = calc_pairwise_distances(tertiary)
        # pad on left and bottom
        tertiary_rank = tf.rank(tertiary).numpy()
        tertiary_padding = tf.repeat(padding, tertiary_rank, axis=0)
        tertiary = tf.pad(tertiary, tertiary_padding, constant_values=tf.cast(padding_value, tertiary.dtype))
        
        # mask transformation
        mask_rank = tf.rank(tertiary_mask).numpy()
        mask_padding = tf.repeat(padding, mask_rank, axis=0)
        tertiary_mask = tf.pad(tertiary_mask, mask_padding, constant_values=0)
        
        # perform crop
        if total_crops > 0:
            batches = create_protein_batches(primary, tertiary, tertiary_mask, stride)
            # transform teritiary to distogram
            for i in range(len(batches)):
                dist_tertiary = to_distogram(batches[i][1], min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
                dist_tertiary = tf.convert_to_tensor(dist_tertiary, dtype=tertiary.dtype)
                dist_tertiary = K.cast_to_floatx(dist_tertiary)
                batches[i] = (batches[i][0], dist_tertiary, batches[i][2])
            return batches
        else:
            tertiary = to_distogram(tertiary, min_val=minimum_bin_val, max_val=maximum_bin_val, num_bins=num_bins)
            tertiary = tf.convert_to_tensor(tertiary, dtype=tertiary.dtype)
            tertiary = K.cast_to_floatx(tertiary)
            return ([(primary, tertiary, tertiary_mask)])
        
if __name__=="__main__":
    path = glob.glob("../proteinnet/data/casp7/training/100/*")
    params = {
    "dim":(64,64), # this is the LxL
    "datasize":None, 
    "features":"primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value":-1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":100,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":False         # if cropping should be random, this has to be implemented later
    }
    dataprovider = DataGenerator(path, **params)
    num_data_points = 0
    for count in dataprovider:
        # print(len(count))
        num_data_points += 1
    print(num_data_points)