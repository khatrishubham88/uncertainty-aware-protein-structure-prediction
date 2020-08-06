import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import expand_dim, calc_pairwise_distances, load_npy_binary, output_to_distancemaps
from utils import random_index
from readData_from_TFRec import parse_tfexample, create_crop2, parse_val_tfexample, parse_test_tfexample
import glob
import copy
import math
import warnings
from tensorflow.python.ops import array_ops
import threading


class DataTestDataGenerator(object):
    def __init__(self, path:list, crop_size=64, category="all", minimum_bin_val=2,
                 maximum_bin_val=22, num_bins=64,
                 batch_size=64):
        self.path = path
        self.crop_size = crop_size
        if category == "all":
            self.category = 5
        elif category == "TBM":
            self.category = 1
        elif category == "TBM-Hard":
            self.category = 3
        elif category == "FM":
            self.category = 2
        elif category == "TBM/Hard":
            self.category = 4
        self.minimum_bin_val = minimum_bin_val
        self.maximum_bin_val = maximum_bin_val
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.raw_dataset = tf.data.TFRecordDataset(self.path)

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, path: list, val_path: list=[None], crop_size=64,
                 datasize=None, features="pri-evo",
                 padding_value=-1, minimum_bin_val=2,
                 maximum_bin_val=22, num_bins=64,
                 batch_size=100, shuffle=False,
                 shuffle_buffer_size=None,
                 random_crop=True, take=None,
                 flattening=True, epochs=1, prefetch = False, experimental_val_take = None,
                 val_shuffle=None, val_shuffle_buffer_size=None, validation_batch_size=None,
                 val_random_crop=None, val_prefetch=None,
                 validation_thinning_threshold=50, training_validation_ratio=None):
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
        self.lock = threading.Lock()
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

        self.make_val_feeder = False
        # make validation dataset feeder
        if val_path[0] is not None:
            self.make_val_feeder = True
            self.val_path = val_path
            self.validation_thinning_threshold = validation_thinning_threshold
            self.experimental_val_take = experimental_val_take
            self.val_idx = []

            self.val_shuffle = val_shuffle
            if self.val_shuffle is None:
                self.val_shuffle = self.shuffle
            self.val_shuffle = False
            self.validation_batch_size = validation_batch_size
            if self.validation_batch_size is None:
                self.validation_batch_size = self.batch_size

            self.val_random_crop = val_random_crop
            if self.val_random_crop is None:
                self.val_random_crop = self.random_crop

            self.val_prefetch = val_prefetch
            if self.val_prefetch is None:
                self.val_prefetch = self.prefetch

            self.val_shuffle_buffer_size = val_shuffle_buffer_size
            if self.val_shuffle_buffer_size is None:
                self.val_shuffle_buffer_size = self.shuffle_buffer_size

            self.validation_raw_dataset = tf.data.TFRecordDataset(self.val_path)
            self.training_validation_ratio = training_validation_ratio
            if self.training_validation_ratio is not None:
                self.validation_datasize = int(self.training_validation_ratio * self.datasize)
                self.val_mask = []

                original_validation_datasize = self.fetch_validation_datasize()
                num_val_repeats = int(np.floor(self.validation_datasize / original_validation_datasize))
                self.validation_datasize = self.make_validation_idx_buffer(num_val_repeats, original_validation_datasize)
                self.validation_raw_dataset = self.validation_raw_dataset.repeat(num_val_repeats)
            else:
                self.validation_datasize = self.fetch_validation_datasize()
            # self.idx_iterator = np.arange(len(self.val_mask))
            number_of_samples = int(np.floor(self.validation_datasize / self.validation_batch_size))*self.validation_batch_size
            count = 0
            idx = 0
            for i in range(len(self.val_mask)):
                if self.val_mask[i]:
                    count +=1
                idx += 1
                if count == number_of_samples:
                    break
            # print("samples = {}, count = {}, idx = {}".format(number_of_samples, count, idx))
            self.idx_iterator = np.arange(idx + 1)
            self.approx_val_sample_per_epoch = idx
            if self.val_shuffle:
                self.validation_raw_dataset = self.validation_raw_dataset.shuffle(self.val_shuffle_buffer_size)

            self.validation_datafeeder = None

            self.val_shape = int(np.floor(self.validation_datasize / self.validation_batch_size))
        else:
            warnings.warn("No path provided! Validation dataset would be ignored!")

        # construct the generator and feeder
        self.datafeeder = None
        # self.construct_feeder()
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
        if self.iterator is None:
            self.construct_feeder()
            self.iterator = self.datafeeder.__iter__()
        return self

    def __next__(self):
        # with self.lock:
        #     output = next(self.iterator)
        #     return output
        output = next(self.iterator)
        return output

    def test_val_size(self):
        count = 0
        for data in self.validation_raw_dataset:
            temp = parse_val_tfexample(data, self.validation_thinning_threshold)
            if all(ret is not None for ret in temp):
                count += 1
        return count

    def get_validation_dataset(self):
        if self.make_val_feeder:
            return self.validation_datafeeder
        else:
            raise ValueError("The validation dataset is not provided!")

    def get_validation_length(self):
        if self.make_val_feeder:
            return self.val_shape
        else:
            raise ValueError("The validation dataset is not provided!")

    def construct_feeder(self):
        def flat_map(features, distogram, mask):
            return (features, distogram, tf.reshape(mask, shape=(-1,)))

        self.datafeeder = tf.data.Dataset.from_generator(self.transformation_generator,
                                                         output_types=(tf.float32, tf.float32, tf.float32),
                                                        #  output_shapes= ((None, None, None, ),
                                                        #                  (None, None, None, ),
                                                        #                  (None, None, ))
                                                         )
        # print(self.datafeeder.element_spec)
        # batch before map, for vectorization.
        self.datafeeder = self.datafeeder.batch(self.batch_size, drop_remainder=True)

        if self.flattening:
            # parallelizing the flattening=
            self.datafeeder = self.datafeeder.map(lambda x, y, z: flat_map(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # no take for validation set, experimental take implemented
        if self.take is not None:
            self.datafeeder = self.datafeeder.take(self.take)

        self.datafeeder = self.datafeeder.repeat(self.epochs)

        # apply prefetch for memory optimization
        if self.prefetch:
            self.datafeeder = self.datafeeder.prefetch(tf.data.experimental.AUTOTUNE)

        if self.make_val_feeder:
            self.validation_datafeeder = tf.data.Dataset.from_generator(self.val_transformation_generator,
                                                         output_types=(tf.float32, tf.float32, tf.float32))

            self.validation_datafeeder = self.validation_datafeeder.batch(self.validation_batch_size, drop_remainder=True)
            if self.flattening:
                self.validation_datafeeder = self.validation_datafeeder.map(lambda x, y, z: flat_map(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if self.experimental_val_take is not None:
                self.validation_datafeeder = self.validation_datafeeder.take(self.experimental_val_take)
            self.validation_datafeeder = self.validation_datafeeder.repeat(self.epochs)
            if self.training_validation_ratio is not None:
                self.idx_iterator = np.tile(self.idx_iterator, self.epochs)
                print("size of idx_iterator = {}".format(self.idx_iterator.shape))
            if self.val_prefetch:
                self.validation_datafeeder = self.validation_datafeeder.prefetch(tf.data.experimental.AUTOTUNE)

    def fetch_datasize(self):
        count = 0
        for _ in self.raw_dataset:
            count +=1
        return count

    def pop_idx(self):
        item, self.idx_iterator = self.idx_iterator[0], self.idx_iterator[1:]
        return int(item)

    def make_validation_idx_buffer(self, repeats, raw_data_size):
        # number of times to repeat the data
        big_data_size = 0
        for c in range(repeats):
            protein_count = 0
            # iterate over raw data of size 96
            tmp_dataset = self.validation_raw_dataset
            for i, data in enumerate(tmp_dataset):
                primary, _, _, _ = parse_val_tfexample(data, self.validation_thinning_threshold)
                if primary is not None:
                    # if the length is less than and equal to crop size only add it once else ignore it
                    if primary.shape[0]<=self.crop_size:
                        # these will be skipped
                        if c > 0:
                            self.val_mask.append(False)
                            self.val_idx.append([None,None])
                        else:
                            self.val_idx.append([0,0])
                            self.val_mask.append(True)
                    else:
                        if self.val_random_crop:
                            x = np.random.randint(0, primary.shape[0] - self.crop_size)
                            y = np.random.randint(0, primary.shape[0] - self.crop_size)
                            if c > 0:
                                retry = True
                                while retry:
                                    if [x,y] in self.val_idx[raw_data_size+i::big_data_size]:
                                        x = np.random.randint(0, primary.shape[0] - self.crop_size)
                                        y = np.random.randint(0, primary.shape[0] - self.crop_size)
                                    else:
                                        retry = False
                                self.val_idx.append([x,y])
                                self.val_mask.append(True)
                            else:
                                big_data_size += 1
                                self.val_idx.append([x,y])
                                self.val_mask.append(True)
                        else:
                            self.val_idx.append([0,0])
                            self.val_mask.append(True)
                    protein_count += 1
        return self.val_mask.count(True)

    def fetch_validation_datasize(self):
        count = 0
        for data in self.validation_raw_dataset:
            temp = parse_val_tfexample(data, self.validation_thinning_threshold)
            if all(ret is not None for ret in temp):
                count +=1
            else:
                continue
        return count

    def transformation_generator(self):
        for data in self.raw_dataset:
            primary, evolutionary, tertiary, ter_mask = parse_tfexample(data)
            transformed_batch = DataGenerator.generator_transform(primary, evolutionary, tertiary, ter_mask,
                                                    features=self.features,
                                                    crop_size=self.crop_size,
                                                    padding_value=self.padding_value,
                                                    minimum_bin_val=self.minimum_bin_val,
                                                    maximum_bin_val=self.maximum_bin_val,
                                                    num_bins=self.num_bins,
                                                    random_crop=self.random_crop,
                                                    transformation_type="training")

            yield transformed_batch # has values (primary, tertiary, tertiary mask)

    def val_transformation_generator(self):
        counter = 0
        for data in self.validation_raw_dataset:
            primary, evolutionary, tertiary, ter_mask = parse_val_tfexample(data, self.validation_thinning_threshold)
            if all(ret is not None for ret in [primary, evolutionary, tertiary, ter_mask]):
                c = counter
                if counter < self.approx_val_sample_per_epoch:
                    with self.lock:
                        counter += 1
                else:
                    with self.lock:
                        counter = 0
                if self.val_mask[c]:
                    transformed_batch = DataGenerator.generator_transform(primary, evolutionary, tertiary, ter_mask,
                                                                        features=self.features,
                                                                        crop_size=self.crop_size,
                                                                        padding_value=self.padding_value,
                                                                        minimum_bin_val=self.minimum_bin_val,
                                                                        maximum_bin_val=self.maximum_bin_val,
                                                                        num_bins=self.num_bins,
                                                                        random_crop=self.val_random_crop,
                                                                        transformation_type="validation",
                                                                        arr_idx = c,
                                                                        val_idx_array = self.val_idx)
                    yield transformed_batch
                else:
                    continue

            else:
                continue

    @staticmethod
    def generator_transform(primary, evolutionary, tertiary, tertiary_mask, features, crop_size, random_crop=True,
                            padding_value=-1, minimum_bin_val=2, maximum_bin_val=22, num_bins=64, transformation_type="training", arr_idx=None, val_idx_array=None):

        # correcting the datatype to avoid errors
        padding_value = float(padding_value)
        minimum_bin_val = float(minimum_bin_val)
        maximum_bin_val = float(maximum_bin_val)
        num_bins = int(num_bins)
        if transformation_type == "training":
            if random_crop:   #this is the case for training data
                index = random_index(primary, crop_size)
            else:             #this is the case for validation data
                index = [0,0]
        elif transformation_type == "validation":
            if arr_idx is None:
                raise ValueError("Index to the validation array cannot be None!")
            index = val_idx_array[arr_idx]

        dist_map = calc_pairwise_distances(tertiary)
        padding_size = math.ceil(primary.shape[0]/crop_size)*crop_size - primary.shape[0]
        # perform cropping + necessary padding
        random_crop = create_crop2(primary, evolutionary, dist_map, tertiary_mask, features, index, crop_size,
                                   padding_value, padding_size, minimum_bin_val, maximum_bin_val, num_bins)
        return random_crop
