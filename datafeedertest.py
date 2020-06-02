from network import ResNet
from dataprovider import DataGenerator
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import glob
from utils import *


def main():
    """
    paths = []
    for i in range(1, 136):
        paths.append('/storage/remote/atcremers45/s0237/casp7/training/100/' + str(i))
    X, mask, y = gather_data_seq_under_limit(paths, 64)
    """
    path = glob.glob("../proteinnet/data/casp7/training/100/*")
    params = {
    "crop_size":64, # this is the LxL
    "datasize":None,
    "features":"primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value":-1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":1,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":True,         # if cropping should be random, this has to be implemented later
    "flattening":True,
    "epochs":1,
    "take":128,

    }
    dataprovider = DataGenerator(path, **params)
    K.clear_session()
    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=16, crop_size=64, dropout_rate=0.15)
    model = nn.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001),
                  loss=masked_categorical_cross_entropy_test())
                #   loss=tf.keras.losses.CategoricalCrossentropy())
    print(model.summary())
    model_hist = model.fit(dataprovider, # (x, y, mask)
                        #    batch_size=16,
                           epochs=10,
                           verbose=2,
                           steps_per_epoch=params.take
                        #    validation_split=0.2
                           )
    print(model_hist.history)
    model.save("model_b16_fs.h5")
    test = model.predict(X[0:32])
    # test = output_to_distancemaps(test, 2, 22, 64)

if __name__=="__main__":
    main()
