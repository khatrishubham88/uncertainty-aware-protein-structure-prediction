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
import sys
sys.setrecursionlimit(100000)
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
    train_path = glob.glob("../proteinnet/data/casp7/training/100/*")
    val_path = glob.glob("../proteinnet/data/casp7/validation/*")
    train_plot = False
    validation_plot = True
    params = {
    "crop_size":64, # this is the LxL
    "datasize":None,
    "features":"pri-evo", # this will decide the number of channel, with primary 20, pri-evo 41
    "padding_value":0, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val":2, # starting bin size
    "maximum_bin_val":22, # largest bin size
    "num_bins":64,         # num of bins to use
    "batch_size":16,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle":True,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":True,         # if cropping should be random, this has to be implemented later
    "flattening":True,
    # "take":8,
    "epochs":30,
    "prefetch": True,
    "val_path": val_path,
    "validation_thinning_threshold": 50,
    "training_validation_ratio": 0.2,
    # "experimental_val_take": 2
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
    if len(dataprovider) <=0:
        raise ValueError("Data reading failed!")

    K.clear_session()
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    if params["features"]=="primary":
        inp_channel = 20
    elif params["features"]=="pri-evo":
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
    else:
        raise ValueError("Wrong Architecture Selected!")

    nn = ResNet(input_channels=inp_channel, output_channels=64, num_blocks=num_blocks, num_channels=num_channels, dilation=[1, 2, 4, 8],
                batch_size=params["batch_size"], crop_size=params["crop_size"], dropout_rate=0.1)
    model = nn.model()
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.06),
                  loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE, global_batch_size=params["batch_size"]))
    tf.print(model.summary())
    


    # to find number of steps for one epoch
    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)

    # to find learning rate patience with minimum 3 and then epoch dependent
    lr_patience = 2

    
    # need to be adjusted for validation loss
    callback_es = tf.keras.callbacks.EarlyStopping('val_loss', verbose=1, patience=5)
    callback_lr = tf.keras.callbacks.ReduceLROnPlateau('val_loss', verbose=1, patience=lr_patience, min_lr=1e-4)
    # to create a new checkpoint directory
    chkpnt_dir = "chkpnt_"
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
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    model.load("minifold_trained/"+archi_style"/saved_model.pb")
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

    model.save_weights(model_dir + "/custom_model_weights_epochs_"+str(params["epochs"])+"_batch_size_"+str(params["batch_size"]))
    model.save(model_dir + '/' + archi_style)
    
    # plot loss
    x_range = range(1,len(model_hist.history["loss"])+1)
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
    plt.plot( x_range, model_hist.history["lr"])
    plt.savefig("learning_rate.png")
    plt.close("all")
    params["epochs"]=1
    dataprovider = DataGenerator(train_path, **params)
    if params.get("val_path", None) is not None:
        validation_data = dataprovider.get_validation_dataset()
        if params.get("experimental_val_take", None) is not None:
            validation_steps = params.get("experimental_val_take", None)
        else:
            validation_steps = dataprovider.get_validation_length()
    if train_plot:
        for j in range(num_of_steps):
            X, y, mask = next(dataprovider)
            # model.save("model_b16_fs.h5")
            mask = mask.numpy()
            y = y.numpy()
            mask = mask.reshape(y.shape[0:-1])
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
            plt.suptitle("Training Data", fontsize=16)
            plt.savefig(result_dir + "/result.png")
            plt.close("all")
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
                plt.suptitle("Training Data", fontsize=16)
                plt.savefig(result_dir + "/result_batch_"+str(j)+"_sample_"+str(i)+".png")
                plt.close("all")

    if validation_plot:
        if params.get("val_path", None) is not None:
            for j, val in enumerate(validation_data):
                X, y, mask = val
                # model.save("model_b16_fs.h5")
                mask = mask.numpy()
                y = y.numpy()
                mask = mask.reshape(y.shape[0:-1])
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
                plt.suptitle("Validation Data", fontsize=16)
                plt.savefig(val_result_dir+"/result.png")
                plt.close("all")

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
                    plt.suptitle("Validation Data", fontsize=16)
                    plt.savefig(val_result_dir + "/result_batch_"+str(j)+"_sample_"+str(i)+".png")
                    plt.close("all")

if __name__=="__main__":
    main()
