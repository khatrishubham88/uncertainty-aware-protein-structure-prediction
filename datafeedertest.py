from network import ResNet
from dataprovider import DataGenerator
import matplotlib.pyplot as plt
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
    "padding_value":0, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":2,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":False,         # if cropping should be random, this has to be implemented later
    "flattening":True,
    "take":16,
    "epochs":100
    }
    dataprovider = DataGenerator(path, **params)
    K.clear_session()
    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=params["batch_size"], crop_size=params["crop_size"], dropout_rate=0.0)
    model = nn.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.003),
                  loss=masked_categorical_cross_entropy_test())
    print(model.summary())
    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)
    callback_es = tf.keras.callbacks.EarlyStopping('loss', verbose=1, patience=10)
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau('loss', verbose=1, patience=5)
    model_hist = model.fit(dataprovider, # (x, y, mask)
                           epochs=params["epochs"],
                           verbose=1,
                           steps_per_epoch = num_of_steps,
                           callbacks=[callback_lr, callback_es]
                           )
    # model = tf.keras.models.load_model('model_b16_fs.h5', compile=False)
    print(model_hist.history)
    print(dataprovider.idx_track)
    # params["take"] = 15
    dataprovider = DataGenerator(path, **params)
    for j in range(params["take"]):
        X, y, mask = next(dataprovider)
        # model.save("model_b16_fs.h5")
        mask = mask.numpy()
        y = y.numpy()
        mask = mask.reshape(y.shape[0:-1])
        # print(y.shape)
        # print(mask[0])
        distance_maps = output_to_distancemaps(y, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
        test = model.predict(X)
        test = output_to_distancemaps(test, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
        plt.figure()
        plt.subplot(131)
        plt.title("Ground Truth")
        plt.imshow(distance_maps[0], cmap='viridis_r')
        plt.subplot(132)
        plt.title("Prediction by model")
        plt.imshow(test[0], cmap='viridis_r')
        plt.subplot(133)
        plt.title("mask")
        plt.imshow(mask[0], cmap='viridis_r')
        plt.savefig("result.png")
        for i in range(params["batch_size"]):
            plt.figure()
            plt.subplot(131)
            plt.title("Ground Truth")
            plt.imshow(distance_maps[i], cmap='viridis_r')
            plt.subplot(132)
            plt.title("Prediction by model")
            plt.imshow(test[i], cmap='viridis_r')
            plt.subplot(133)
            plt.title("mask")
            plt.imshow(mask[i], cmap='viridis_r')
            plt.savefig("result_batch_"+str(j)+"_sample_"+str(i)+".png")


if __name__=="__main__":
    main()
