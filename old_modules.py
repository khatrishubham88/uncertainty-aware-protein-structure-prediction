import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import expand_dim, calc_pairwise_distances, to_distogram, load_npy_binary, output_to_distancemaps
from utils import pad_mask, pad_primary, pad_tertiary, masked_categorical_cross_entropy
from readData_from_TFRec import widen_seq, parse_tfexample, create_protein_batches
from readData_from_TFRec import widen_seq, parse_dataset, NUM_AAS, NUM_EVO_ENTRIES, NUM_DIMENSIONS
import math

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

class DataGeneratorOld(object):
    'Generates data for Keras'
    def __init__(self, path: list, dim=(64,64), datasize=None, features="primary", 
                 padding_value=-1, minimum_bin_val=2, 
                 maximum_bin_val=22, num_bins=64, 
                 batch_size=100, shuffle=False, 
                 shuffle_buffer_size=None, random_crop=False, take=None, flattening=False):
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
            transformed_batch = DataGeneratorOld.generator_transform(primary, evolutionary, tertiary, ter_mask, 
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

