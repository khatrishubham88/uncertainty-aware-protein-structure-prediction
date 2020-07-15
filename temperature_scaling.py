import glob
import os

import matplotlib.pyplot as plt

from dataprovider import DataGenerator
from utils_temperature_scaling import *


if __name__ == '__main__':
    train_path = glob.glob("P:/casp7/casp7/training/100/*")
    val_path = glob.glob("P:/casp7/casp7/validation/1")

    params = {
        "crop_size": 64,  # this is the LxL
        "datasize": None,
        "features": "pri-evo",  # this will decide the number of channel, with primary 20, pri-evo 41
        "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
        "minimum_bin_val": 2,  # starting bin size
        "maximum_bin_val": 22,  # largest bin size
        "num_bins": 64,  # num of bins to use
        "batch_size": 128,  # batch size for training, check if this is needed here or should be done directly in fit?
        "shuffle": True,  # if wanna shuffle the data, this is not necessary
        "shuffle_buffer_size": None,  # if shuffle is on size of shuffle buffer, if None then =batch_size
        "random_crop": True,  # if cropping should be random, this has to be implemented later
        "flattening": True,
        "epochs": 1,
        "prefetch": True,
        "val_path": val_path,
        "validation_thinning_threshold": 50,
        "training_validation_ratio": 0.2,
    }

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

    # to find number of steps for one epoch
    try:
        num_of_steps = params["take"]
    except:
        num_of_steps = len(dataprovider)

    num_blocks = [28]
    num_channels = [64]
    dilation = [1, 2, 4, 8]
    dropout_rate = 0.1
    reg_strength = 0.
    weights_path = "P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16"

    model = model_with_logits_output(features=params["features"], model_weights=weights_path,
                                     output_channels=params["num_bins"], num_blocks=num_blocks,
                                     num_channels=num_channels, dilation=dilation, batch_size=params["batch_size"],
                                     crop_size=params["crop_size"], dropout_rate=dropout_rate,
                                     reg_strength=reg_strength)
    #temperatures = [[1.0 for i in range(64)]]
    temperatures = [1.0]
    ece_before = []
    ece_after = []
    if params.get("val_path", None) is not None:
        for j, val in enumerate(validation_data):
            X, ground_truth, mask = val
            mask = mask.numpy()
            ground_truth = ground_truth.numpy()
            output = model.predict(X, batch_size=params["batch_size"])
            confids = softmax(output, axis=3)
            labels = np.argmax(ground_truth, axis=3)
            preds = np.argmax(confids, axis=3)
            confs = np.max(confids, axis=3)

            confs = np.reshape(confs, -1)
            preds = np.reshape(preds, -1)
            mask = np.reshape(mask, -1)
            labels = np.reshape(labels, -1)

            accuracies, confidences, bin_lengths = get_bin_info(confs, preds, labels, mask, bin_size=0.1)
            gaps = np.abs(np.array(accuracies) - np.array(confidences))
            ece = expected_calibration_error(confs, preds, labels, mask)
            ece_before.append(ece)
            print(str(j) + ". ECE: " + str(ece))

            if j % 5 == 0:
                print("Before Temperature Scaling:")
                print("Accuracies: " + str(accuracies))
                print("Confidences: " + str(confidences))
                print("Gaps: " + str(gaps))

            X, ground_truth, mask = val
            mask = mask.numpy()
            ground_truth = ground_truth.numpy()
            output = model.predict(X, batch_size=params["batch_size"])
            output = np.reshape(output, (-1, 64))
            mask = np.reshape(mask, (-1))
            if j % 5 != 0:
                if j == 0:
                    history, T = calibrate_temperature(output, labels, mask)
                else:
                    history, T = calibrate_temperature(output, labels, mask, temp=tf.Variable(temperatures[-1]))
                    print(str(j) + ". Temperature: " + str(T))
                temperatures.append(T)
            else:
                print(str(j) + ". Predicting with learned temperature..")

            X, ground_truth, mask = val
            mask = mask.numpy()
            ground_truth = ground_truth.numpy()
            logits = model.predict(X, batch_size=params["batch_size"])
            if j != 0:
                confids_scaled = predict_with_temperature(logits, temp=temperatures[-1])
            else:
                confids_scaled = predict_with_temperature(logits, temp=temperatures[0])
            labels_scaled = np.argmax(ground_truth, axis=3)
            preds_scaled = np.argmax(confids_scaled, axis=3)
            confs_scaled = np.max(confids_scaled, axis=3)

            confs_scaled = np.reshape(confs_scaled, -1)
            preds_scaled = np.reshape(preds_scaled, -1)
            mask = np.reshape(mask, -1)
            labels_scaled = np.reshape(labels_scaled, -1)

            accuracies_scaled, confidences_scaled, bin_lengths_scaled = get_bin_info(confs_scaled, preds_scaled,
                                                                                     labels_scaled, mask, bin_size=0.1)
            gaps_scaled = np.abs(np.array(accuracies_scaled) - np.array(confidences_scaled))
            ece_scaled = expected_calibration_error(confs_scaled, preds_scaled, labels_scaled, mask)
            ece_after.append(ece_scaled)
            print(str(j) + ". ECE: " + str(ece_scaled))
            if j % 5 != 0:
                print("================================")
            if j % 5 == 0:
                print("After Temperature Scaling:")
                print("Accuracies: " + str(accuracies_scaled))
                print("Confidences: " + str(confidences_scaled))
                print("Gaps: " + str(gaps_scaled))
                print("================================")

                """
                plt.style.use('ggplot')
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(22.5, 4), sharex='col', sharey='row')
                rel_diagram_sub(accuracies, confidences, ax[0])
                rel_diagram_sub(accuracies_scaled, confidences_scaled, ax[1])
                plt.show()
                """

    temperatures_dir = "temperatures"
    if os.path.isdir(temperatures_dir) is False:
        os.mkdir(temperatures_dir)
    name_temperature_binary_file = temperatures_dir + "/" + "temperatures"
    np.save(name_temperature_binary_file, temperatures[-1])

    print(sum(ece_before[1:]) / len(ece_before[1:]))
    print(sum(temperatures[1:]) / len(temperatures[1:]))
    print(sum(ece_after[1:]) / len(ece_after[1:]))
