from network_sparse import ResNet, ResNetV2
# from dataprovider_sparse import DataGenerator
from dataprovider import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import glob
import os
import time
import warnings
from utils import accuracy_metric, precision_metric
import sys

from utils import *
import matplotlib
matplotlib.use('Agg')
plt.style.use("ggplot")
sys.setrecursionlimit(100000)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")
# tf.autograph.set_verbosity(0)

def main():
    train_path = glob.glob("../proteinnet/data/casp7/training/100/*")
    val_path = glob.glob("../proteinnet/data/casp7/validation/1")
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
    "shuffle":False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size":None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop":True,         # if cropping should be random, this has to be implemented later
    "val_random_crop":True,
    "flattening":True,
    #"take":8,
    "epochs":3,
    "prefetch": True,
    "val_path": val_path,
    "validation_thinning_threshold": 50,
    "training_validation_ratio": 0.05,
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
    elif archi_style == "test_small_block":
        print("Running on test architecture")
    else:
        print("It is a wrong architecture!")

    # printing the above params for rechecking
    print("Logging the parameters used")
    for k, v in params.items():
        print("{} = {}".format(k,v))
    # time.sleep(20)

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
                     num_channels=num_channels, dilation=[1, 2, 4, 8], batch_size=params["batch_size"],
                     crop_size=params["crop_size"], dropout_rate=0.15, reg_strength=1e-4, logits=False, sparse=False,
                     kernel_initializer="he_normal", kernel_regularizer="l2", mc_dropout=True,
                     mc_sampling=5)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.006),
                  loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE, global_batch_size=params["batch_size"]))
    model.load_weights("minifold_trained/custom_model_weights_epochs_30_batch_size_16").expect_partial()
    tf.print(model.summary())
    del dataprovider
    dataprovider = DataGenerator(train_path, **params)
    if params.get("val_path", None) is not None:
        validation_data = dataprovider.get_validation_dataset()
        if params.get("experimental_val_take", None) is not None:
            validation_steps = params.get("experimental_val_take", None)
        else:
            validation_steps = dataprovider.get_validation_length()
    if validation_plot:
        if params.get("val_path", None) is not None:
            data_acc = []
            mean_acc = []
            for j, val in enumerate(validation_data):
                X, y, mask = val
                # model.save("model_b16_fs.h5")
                mask = mask.numpy()
                y = y.numpy()
                mask = mask.reshape(y.shape[0:-1])
                # mc predict
                mc_pred, mean_predict = model.mc_predict(X, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                acc = mc_accuracy(y, mc_pred, mask)
                # print(mc_pred.shape)
                data_acc.append(acc)
                mean_sample_acc = mc_accuracy(y, mean_predict, mask)
                mean_acc.append(mean_sample_acc)
                # batch plotting
                reshaped_acc = tf.convert_to_tensor(acc, dtype=tf.float32)
                reshaped_acc = tf.transpose(reshaped_acc, perm=[1, 0])
                y_true_dmap = output_to_distancemaps(y, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                mean_predict_dmap = output_to_distancemaps(mean_predict, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                mc_pred = tf.transpose(mc_pred, perm=[1,0,2,3,4])
                for i in range(int(reshaped_acc.shape[0])):
                    mc_hist_plot(val_result_dir+"/val_data_hist_batch_"+str(j)+"_sample_"+str(i)+".png", reshaped_acc[i].numpy(), mean_sample_acc[i], "Accuracy Distribution Based on Sampling")
                    best_idx = np.argmax(reshaped_acc[i].numpy())
                    # distance_map_plotter(val_result_dir+"/val_data_distmap_batch_"+str(j)+"_sample_"+str(i)+"_mean.png", y_true_dmap[i], mean_predict_dmap[i], mask[i], title="Distancemap Plots For Validation Set")
                    best_pred = output_to_distancemaps(mc_pred[i, best_idx], params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                    # print("input shape = {}, output shpe = {}".format(mc_pred[i, best_idx].shape, best_pred.shape))
                    # distance_map_plotter(val_result_dir+"/val_data_distmap_batch_"+str(j)+"_sample_"+str(i)+"_best.png", y_true_dmap[i], best_pred, mask[i], title="Distancemap Plots For Validation Set")
                    mc_distance_map_plotter(val_result_dir+"/val_data_distmap_batch_"+str(j)+"_sample_"+str(i)+".png", y_true_dmap[i], mean_predict_dmap[i], best_pred, mask[i], title="Distancemap Plots For Validation Set")
                    if i%8==0:
                        plt.close("all")
                plt.close("all")
            arr = np.array(data_acc)
            mean_arr = np.array(mean_acc)
            print(arr)
            try:
                print("Writing numpy binary!")
                np.save("mcd_sampling_acc.npy", arr)
                np.save("mcd_mean_acc.npy", mean_arr)
            except:
                print("Writing numpy CSV!")
                if os.path.isfile("mcd.npy"):
                    os.remove("mcd.npy")
                arr2 = arr.transpose(1,0,2)
                arr2 = arr2.reshape(arr2.shape[0],-1)
                np.savetxt("mcd.csv", arr2)
                

if __name__=="__main__":
    main()
