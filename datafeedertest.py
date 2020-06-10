from network import ResNet
from dataprovider import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import glob
import os
import time
import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# warnings.filterwarnings("ignore")
# tf.autograph.set_verbosity(0)
from utils import *
# tf.config.experimental_run_functions_eagerly(True)

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
    "take":8,
    "epochs":2,
    "prefetch": True
    }
    print("Logging the parameters used")
    for k, v in params.items():
        print("{} = {}".format(k,v))
    #time.sleep(60)    
    result_dir = "test_results"
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
    dataprovider = DataGenerator(path, **params)
    print("Total Dataset size = {}".format(len(dataprovider)))
    if len(dataprovider) <=0:
        raise ValueError("Data reading failed!")
    K.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=params["batch_size"], crop_size=params["crop_size"], dropout_rate=0.1)
    model = nn.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.003),
                loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE, global_batch_size=params["batch_size"]))
                # loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE))
    tf.print(model.summary())
    #loss = 89064.46875
    
    print(model.layers[1].kernel_initializer.scale)
    
    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)

    if int(params["epochs"]/10) <= 3:
        lr_patience = 3
    else:
        lr_patience = int(params["epochs"]/10)
    
    callback_es = tf.keras.callbacks.EarlyStopping('loss', verbose=1, patience=5)
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau('loss', verbose=1, patience=lr_patience)
    model_hist = model.fit(dataprovider, # (x, y, mask)
                           epochs=params["epochs"],
                           verbose=1,
                           steps_per_epoch = num_of_steps,
                           callbacks=[callback_lr, callback_es]
                           )
    # model = tf.keras.models.load_model('model_b16_fs.h5', compile=False)
    print(model_hist.history)   
    # print(dataprovider.idx_track)
    # params["take"] = 15
    dataprovider = DataGenerator(path, **params)
    
    # plot loss
    x_range = range(1,params["epochs"]+1)
    # print(list(x_range))
    plt.figure()
    plt.title("Loss plot")
    plt.plot(x_range, model_hist.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    plt.figure()
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.plot( x_range, model_hist.history["lr"])
    plt.savefig("learning_rate.png")
    
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
            plt.savefig(result_dir + "/result_batch_"+str(j)+"_sample_"+str(i)+".png")
    model.save_weights("custom_model_weights_epochs_"+str(params["epochs"])+"_batch_size_"+str(params["batch_size"]))


if __name__=="__main__":
    main()
