import argparse
import glob
import pickle
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from dataprovider import DataGenerator
from network import ResNetV2
from tensorflow.keras.metrics import CategoricalAccuracy
from utils import *

sys.setrecursionlimit(100000)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")
# tf.autograph.set_verbosity(0)


def train(traindata_path, valdata_path, classweight_path, val_thinning):
    train_path = glob.glob(traindata_path)
    val_path = glob.glob(valdata_path)
    class_weights = classweight_path
    val_thinning = int(val_thinning)

    with open(class_weights, 'rb') as handle:
        cws = pickle.load(handle)

    params = {
    "crop_size":64, # this is the LxL
    "datasize":None,
    "features":"pri-evo", # this will decide the number of channel, with primary 20, pri-evo 41
    "padding_value":0, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":64,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":True,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":True,         # if cropping should be random, this has to be implemented later
    "val_random_crop":True,
    "flattening":True,
    "epochs":30,
    "prefetch": True,
    "val_path": val_path,
    "validation_thinning_threshold": val_thinning,
    "training_validation_ratio": 0.2
    }
    archi_style = "one_group"
    # archi_style = "two_group_prospr"
    # archi_style = "two_group_alphafold"
    if archi_style=="one_group":
        print("Training on Minifold architecture!")
    elif archi_style == "two_group_prospr":
        print("Training on ProSPR architecture!")
    elif archi_style == "two_group_alphafold":
        print("Training on Alphafold architecture!")
    elif archi_style == "test_small_block":
        print("Running on test architecture")
    else:
        print("It is a wrong architecture!")

    # printing the above params for rechecking
    print("Logging the parameters used")
    for k, v in params.items():
        print("{} = {}".format(k,v))
    time.sleep(20)

    # setting up directory to add results after training
    result_dir = "test_results"
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
    if params.get("val_path", None) is not None:
        val_result_dir = result_dir + "/val_data"
        if os.path.isdir(val_result_dir) is False:
            os.mkdir(val_result_dir)
    # instantiate data provider
    dataprovider = DataGenerator(train_path, **params)

    print("Total Dataset size = {}".format(len(dataprovider)))

    if params.get("val_path", None) is not None:
        validation_data = dataprovider.get_validation_dataset()
        validation_steps = dataprovider.get_validation_length()
        print("Validation Dataset size = {}".format(validation_steps))

    # this is just for experimenting
    if params.get("experimental_val_take", None) is not None:
        validation_steps = params.get("experimental_val_take", None)
        print("Experimenting on validation Dataset size = {}".format(validation_steps))

    # if path is wrong this will throw error
    if len(dataprovider) <= 0:
        raise ValueError("Data reading failed!")

    K.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    if params["features"] == "primary":
        inp_channel = 20
    elif params["features"] == "pri-evo":
        inp_channel = 41

    if archi_style == "one_group":
        num_blocks = [28]
        num_channels = [64]
    elif archi_style == "two_group_prospr":
        num_blocks = [28, 192]
        num_channels = [128, 64]
    elif archi_style == "two_group_alphafold":
        num_blocks = [28, 192]
        num_channels = [256, 128]
    elif archi_style == "test_small_block":
        num_blocks = [8]
        num_channels = [64]
    else:
        raise ValueError("Wrong Architecture Selected!")

    model = ResNetV2(input_channels=inp_channel, output_channels=params["num_bins"], num_blocks=num_blocks,
                     num_channels=num_channels,
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     dropout_rate=0.1, reg_strength=1e-4, logits=True, sparse=False, kernel_initializer="he_normal",
                     kernel_regularizer="l2", mc_dropout=False, class_weights=cws)
    # model = nn.model()
    print(type(model))

    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.006, clipnorm=1.0),
                  loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE, global_batch_size=params["batch_size"]), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    tf.print(model.summary())

    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)

    # to find learning rate patience with minimum 3 and then epoch dependent
    lr_patience = 3

    callback_es = tf.keras.callbacks.EarlyStopping('loss', verbose=1, patience=5)
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau('loss', verbose=2, patience=lr_patience)

    # to create a new checkpoint directory
    chkpnt_dir = "class_weight_chkpnt_"
    suffix = 1
    while True:
        chkpnt_test_dir = chkpnt_dir + str(suffix)
        if os.path.isdir(chkpnt_test_dir):
            suffix += 1
            chkpnt_test_dir = chkpnt_dir + str(suffix)
        else:
            chkpnt_dir = chkpnt_test_dir
            os.mkdir(chkpnt_dir)
            break
    checkpoint_path = chkpnt_dir + "/chkpnt"
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')

    if params.get("val_path", None) is not None:
        model_hist = model.fit(dataprovider, # (x, y, mask)
                            epochs=params["epochs"],
                            verbose=1,
                            steps_per_epoch=num_of_steps,
                            validation_data=validation_data,
                            validation_steps=validation_steps,
                            callbacks=[callback_lr, callback_es, callback_checkpoint]
                            )
    else:
        model_hist = model.fit(dataprovider, # (x, y, mask)
                            epochs=params["epochs"],
                            verbose=1,
                            steps_per_epoch=num_of_steps,
                            callbacks=[callback_lr, callback_es]
                            )
    print(model_hist.history)

    model_dir = "model_weights"
    if os.path.isdir(model_dir) is False:
        os.mkdir(model_dir)

    model.save_weights(model_dir + "/class_weight_model_weights_epochs_"+str(params["epochs"])+"_batch_size_"+str(params["batch_size"]))
    model.save(model_dir + '/' + archi_style)

    # plot loss
    x_range = range(1,len(model_hist.history["loss"]) + 1)
    plt.figure()
    plt.title("Loss plot")
    plt.plot(x_range, model_hist.history["loss"], label="Training loss")
    if params.get("val_path", None) is not None:
        plt.plot(x_range, model_hist.history["val_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.figure()
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.plot(x_range, model_hist.history["lr"])
    plt.savefig("learning_rate.png")
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument("--traindata_path", help="Path to train set e.g. /path/to/train")
    parser.add_argument("--valdata_path", help="Path to validation set e.g. /path/to/val")
    parser.add_argument("--classweight_path", help="Path to model weights e.g. /path/to/class_weights.pickle")
    parser.add_argument("--val_thinning", help="Validation thinning equal to 30, 50, 70, 90, 95 or 100")

    args = parser.parse_args()
    traindata_path = args.traindata_path
    valdata_path = args.valdata_path
    classweight_path = args.classweight_path
    val_thinning = args.val_thinning

    train(traindata_path=traindata_path, valdata_path=valdata_path, classweight_path=classweight_path,
          val_thinning=val_thinning)