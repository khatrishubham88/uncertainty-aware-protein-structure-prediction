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
from clr_callback import CyclicLR
# tf.config.experimental_run_functions_eagerly(True)


def main():
    train_path = glob.glob("P:/casp7/casp7/training/100/*")
    val_path = glob.glob("P:/casp7/casp7/validation/1")
    train_plot = True
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
    else:
        raise ValueError("Wrong Architecture Selected!")

    clr = CyclicLR(
        mode="triangular2",
        base_lr=1e-7,
        max_lr=1e-3,
        step_size=8*2159)

    nn = ResNet(input_channels=inp_channel, output_channels=64, num_blocks=num_blocks, num_channels=num_channels,
                dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"], dropout_rate=0.1,
                reg_strength=0.)
    model = nn.model()
    model.load_weights("P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16").expect_partial()
    model.compile(loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE,
                                                             global_batch_size=params["batch_size"]))

    tf.print(model.summary())

    # to find number of steps for one epoch
    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)

    # to find learning rate patience with minimum 3 and then epoch dependent
    # lr_patience = 2
    # need to be adjusted for validation loss
    # callback_es = tf.keras.callbacks.EarlyStopping('val_loss', verbose=1, patience=5)
    # callback_lr = tf.keras.callbacks.ReduceLROnPlateau('val_loss', verbose=1, patience=lr_patience, min_lr=1e-4)
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
    # model.load("minifold_trained/"+archi_style"/saved_model.pb")
    if params.get("val_path", None) is not None:
        history = model.fit(dataprovider, # (x, y, mask)
                            epochs=params["epochs"],
                            verbose=1,
                            steps_per_epoch=num_of_steps,
                            validation_data=validation_data,
                            validation_steps=validation_steps,
                            callbacks=[clr, callback_checkpoint]
                            )
    else:
        history = model.fit(dataprovider, # (x, y, mask)
                            epochs=params["epochs"],
                            verbose=1,
                            steps_per_epoch=num_of_steps,
                            callbacks=[clr, callback_checkpoint]
                            )
    print(history.history)

    model_dir = "model_weights"
    if os.path.isdir(model_dir) is False:
        os.mkdir(model_dir)

    model.save_weights(model_dir + "/custom_model_weights_epochs_"+str(params["epochs"])+"_batch_size_"+str(params["batch_size"]))
    model.save(model_dir + '/' + archi_style)

    """
    history = {'loss': [0.16137753427028656, 0.1508493274450302, 0.1491297036409378, 0.14445514976978302,
                        0.14317423105239868, 0.14306320250034332, 0.14252369105815887, 0.14263318479061127,
                        0.1409928947687149, 0.14060086011886597, 0.1409199982881546, 0.14049330353736877,
                        0.14039155840873718, 0.14001856744289398, 0.14040112495422363, 0.14005810022354126],
               'val_loss': [0.14852797985076904, 0.18629467487335205, 0.25842994451522827, 0.13753165304660797,
                            0.13571062684059143, 0.13480843603610992, 0.13543257117271423, 0.14535877108573914,
                            0.13290998339653015, 0.133968785405159, 0.13277876377105713, 0.13283240795135498,
                            0.13402500748634338, 0.13362358510494232, 0.13353055715560913, 0.1329270601272583],
               'lr': [0.06, 0.06, 0.06, 0.006, 0.006, 0.006, 0.006, 0.006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006,
                      1e-04, 1e-04, 1e-04]}
    """

    # plot loss
    x_range = range(1, len(history.history["loss"]) + 1)
    plt.figure()
    plt.title("Loss plot")
    plt.plot(x_range, history["loss"], label="Training loss")
    if params.get("val_path", None) is not None:
        plt.plot(x_range, history["val_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.figure()
    plt.title("Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.plot(x_range, history["lr"])
    plt.savefig("learning_rate.png")
    plt.close("all")

    """
    model.load_weights("P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16").expect_partial()

    dataprovider = DataGenerator(val_path, **params)
    if params.get("train_path", None) is not None:
        validation_data = dataprovider.get_validation_dataset()
        if params.get("experimental_val_take", None) is not None:
            validation_steps = params.get("experimental_val_take", None)
        else:
            validation_steps = dataprovider.get_validation_length()
    if train_plot:
        for j in range(num_of_steps):
            X, y, mask = next(dataprovider)
            mask = mask.numpy()
            y = y.numpy()
            mask = mask.reshape(y.shape[0:-1])
            distance_maps = output_to_distancemaps(y, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
            contact_maps_true = contact_map_from_distancemap(distance_maps)
            test = model.predict(X)
            test = output_to_distancemaps(test, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
            contact_maps_pred = contact_map_from_distancemap(test)
            plt.figure()
            plt.subplot(141)
            plt.title("Ground Truth")
            plt.imshow(distance_maps[0], cmap='viridis_r')
            plt.subplot(142)
            plt.title("Prediction")
            plt.imshow(test[0], cmap='viridis_r')
            plt.subplot(143)
            plt.title("CM True")
            plt.imshow(contact_maps_true[0], cmap='viridis_r')
            plt.subplot(144)
            correct = ((contact_maps_pred[0] == contact_maps_true[0]) * mask[0]).sum() / np.count_nonzero(mask[0])
            plt.title("CM Pred: " + str(round(correct, 4) * 100) + "%")
            plt.imshow(contact_maps_pred[0], cmap='viridis_r')
            plt.suptitle("Training Data", fontsize=16)
            plt.savefig(result_dir + "/result.png")
            plt.close("all")
            for i in range(params["batch_size"]):
                plt.figure()
                plt.subplot(141)
                plt.title("Ground Truth")
                plt.imshow(distance_maps[i], cmap='viridis_r')
                plt.subplot(142)
                plt.title("Prediction")
                plt.imshow(test[i], cmap='viridis_r')
                plt.subplot(143)
                plt.title("CM True")
                plt.imshow(contact_maps_true[i], cmap='viridis_r')
                plt.subplot(144)
                correct = ((contact_maps_pred[i] == contact_maps_true[i]) * mask[i]).sum() / np.count_nonzero(mask[i])
                plt.title("CM Pred: " + str(round(correct, 4) * 100) + "%")
                plt.imshow(contact_maps_pred[i], cmap='viridis_r')
                plt.suptitle("Training Data", fontsize=16)
                plt.savefig(result_dir + "/result_batch_"+str(j)+"_sample_"+str(i)+".png")
                plt.close("all")

    if validation_plot:
        if params.get("val_path", None) is not None:
            for j, val in enumerate(validation_data):
                X, y, mask = val
                mask = mask.numpy()
                y = y.numpy()
                mask = mask.reshape(y.shape[0:-1])
                distance_maps = output_to_distancemaps(y, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                contact_maps_true = contact_map_from_distancemap(distance_maps)
                test = model.predict(X)
                accuracies, _ = accuracy_metric(y, test, mask)
                precisions, _ = precision_metric(y, test, mask)
                recalls, _ = recall_metric(y, test, mask)
                test = output_to_distancemaps(test, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
                contact_maps_pred = contact_map_from_distancemap(test)
                plt.figure()
                plt.subplot(141)
                plt.title("Ground Truth")
                plt.imshow(distance_maps[0], cmap='viridis_r')
                plt.subplot(142)
                plt.title("Prediction")
                plt.imshow(test[0], cmap='viridis_r')
                plt.subplot(143)
                plt.title("CM True")
                plt.imshow(contact_maps_true[0], cmap='viridis_r')
                plt.subplot(144)
                plt.title("CM Pred")
                plt.imshow(contact_maps_pred[0], cmap='viridis_r')
                plt.suptitle("Acc: " + str(round(accuracies[0] * 100, 2)) + "%, Prec: " + str(
                    round(precisions[0] * 100, 2)) + "%, Rec: " + str(
                    round(recalls[0] * 100, 2)) + "%, F1-Score: " + str(
                    round(f_beta_score(precisions[0], recalls[0], beta=1) * 100, 2)) + "%",
                             fontsize=12)
                plt.savefig(val_result_dir+"/result.png")
                plt.close("all")
                for i in range(params["batch_size"]):
                    plt.figure()
                    plt.subplot(141)
                    plt.title("Ground Truth")
                    plt.imshow(distance_maps[i], cmap='viridis_r')
                    plt.subplot(142)
                    plt.title("Prediction")
                    plt.imshow(test[i], cmap='viridis_r')
                    plt.subplot(143)
                    plt.title("CM True")
                    plt.imshow(contact_maps_true[i], cmap='viridis_r')
                    plt.subplot(144)
                    plt.title("CM Pred")
                    plt.suptitle("Acc: " + str(round(accuracies[i] * 100, 2)) + "%, Prec: " + str(
                        round(precisions[i] * 100, 2)) + "%, Rec: " + str(
                        round(recalls[i] * 100, 2)) + "%, F1-Score: " + str(
                        round(f_beta_score(precisions[i], recalls[i], beta=1) * 100, 2)) + "%",
                                 fontsize=12)
                    plt.imshow(contact_maps_pred[i], cmap='viridis_r')
                    plt.savefig(val_result_dir + "/result_batch_"+str(j)+"_sample_"+str(i)+".png")
                    plt.close("all")
    """

if __name__=="__main__":
    main()
