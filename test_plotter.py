from network import ResNetV2
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
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm, parse_dataset
import argparse
import glob
import random
import sys
from utils_temperature_scaling import *
from utils import *

params = {
    "crop_size": 64,  # this is the LxL
    "datasize": None,
    "features": "pri-evo",  # this will decide the number of channel, with primary 20, pri-evo 41
    "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val": 2,  # starting bin size
    "maximum_bin_val": 22,  # largest bin size
    "num_bins": 64,  # num of bins to use
    "batch_size": 1,  # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle": False,  # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size": None,  # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop": True,  # if cropping should be random, this has to be implemented later
    "val_random_crop": True,
    "flattening": True,
}


def plotter(testdata_path, modelling_group, model_path, result_dir):
    testdata_path = glob.glob(testdata_path + '/*')
    modelling_group = int(modelling_group)
    num_blocks = [28]
    num_channels = [64]
    if params["features"] == "primary":
        inp_channel = 20
    elif params["features"] == "pri-evo":
        inp_channel = 41
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
    X = []
    y = []
    mask = []
    print(testdata_path)
    for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path, modelling_group):
        if primary is not None:
            primary_2D = widen_seq(primary)
            pssm = widen_pssm(evolutionary)
            dist_map = calc_pairwise_distances(tertiary)
            padding_size = math.ceil(primary.shape[0] / params["crop_size"]) * params["crop_size"] - primary.shape[0]
            padded_primary = pad_feature2(primary_2D, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_evol = pad_feature2(pssm, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_dist_map = pad_feature2(dist_map, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_mask = pad_feature2(ter_mask, params["crop_size"], params["padding_value"], padding_size, 2)
            crops = create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask,
                                           params["crop_size"], params["crop_size"])

            for crop in crops:
                X.append(crop[0])  # batch[0] of type eager tensor
                y.append(crop[1])  # batch[1] of type nd-array
                mask.append(crop[2])  # batch[2] of type eager tensor
    print('Finish data extraction..')
    print('Begin model evaluation...')
    X = tf.convert_to_tensor(X)
    y = np.asarray(y)
    mask = tf.convert_to_tensor(mask)
    mask = np.asarray(mask)
    if X.shape[0] % params['batch_size'] != 0:
        drop_samples = X.shape[0] - ((X.shape[0] // params['batch_size']) * params['batch_size'])
        X = X[0:X.shape[0] - drop_samples, :, :]
        mask = mask[0:mask.shape[0] - drop_samples, :, :]
        y = y[0:y.shape[0] - drop_samples, :, :, :]
    model = ResNetV2(input_channels=inp_channel, output_channels=params["num_bins"], num_blocks=num_blocks,
                     num_channels=num_channels,
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     dropout_rate=0.15, reg_strength=1e-4, logits=False, sparse=False, kernel_initializer="he_normal",
                     kernel_regularizer="l2", mc_dropout=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.006, clipnorm=1.0),
                  loss=CategoricalCrossentropyForDistributed(reduction=tf.keras.losses.Reduction.NONE,
                                                             global_batch_size=params["batch_size"]),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.load_weights(model_path).expect_partial()

    output = model.predict(X, verbose=1, batch_size=params["batch_size"])
    y_predict = output_to_distancemaps(output, 2, 22, 64)
    gt = output_to_distancemaps(y, 2, 22, 64)

    accuracy, precision, recall, f1, _ = distogram_metrics(y, output, mask, params['minimum_bin_val'],
                                                           params['maximum_bin_val'], params['num_bins'])
    print('Distogram based Accuracy:', accuracy)
    print('Distogram based Precision:', precision)
    print('Distogram based Recall:', recall)
    print('Distogram based F1-score:', f1)

    entropy = entropy_func(output)
    print('Prediction Entropy:', entropy)

    contact_maps_pred = contact_map_from_distancemap(y_predict)
    contact_maps_true = contact_map_from_distancemap(gt)
    sample_accuracies, cm_accuracy = accuracy_metric(y, output, mask)
    sample_precisions, cm_precision = precision_metric(y, output, mask)
    sample_recalls, cm_recall = recall_metric(y, output, mask)
    cm_fscore = f_beta_score(cm_precision, cm_recall, beta=1)

    print("Contact Map Accuracy: " + str(cm_accuracy))
    print("Contact Map Precision: " + str(cm_precision))
    print("Contact Map Recall: " + str(cm_recall))
    print("Contact Map F1-Score: " + str(cm_fscore))

    for i in range(y_predict.shape[0]):
        plt.figure()
        plt.subplot(141)
        plt.title("Ground Truth")
        plt.imshow(gt[i], cmap='viridis_r')
        plt.subplot(142)
        plt.title("Prediction")
        plt.imshow(y_predict[i], cmap='viridis_r')
        plt.subplot(143)
        plt.title("CM True")
        plt.imshow(contact_maps_true[i], cmap='viridis_r')
        plt.subplot(144)
        plt.title("CM Pred")
        plt.imshow(contact_maps_pred[i], cmap='viridis_r')
        accuracy, precision, recall, f1, _ = distogram_metrics(y[i], output[i], mask[i], params['minimum_bin_val'],
                                                               params['maximum_bin_val'],
                                                               params['num_bins'], single_sample=True)
        plt.suptitle("Acc: " + str(round(sample_accuracies[i] * 100, 2)) + "%, Prec: " + str(
            round(sample_precisions[i] * 100, 2)) + "%, Rec: " + str(
            round(sample_recalls[i] * 100, 2)) + "%, F1-Score: " + str(
            round(f_beta_score(sample_accuracies[i], sample_recalls[i], beta=1) * 100, 2)) + "%,\n"
                     + "Distogram Acc: " + str(
            round(accuracy * 100, 2)) + "%, Prec: " + str(
            round(precision * 100, 2)) + "%, Rec: " + str(
            round(recall * 100, 2)) + "%, F1-Score: " + str(
            round(f1 * 100, 2)), fontsize=12)
        plt.savefig(result_dir + "/result" + str(i) + ".png")
        if i % 8 == 0:
            plt.close("all")



        """
        classes = [i + 0 for i in range(64)]
        title = "Confusion matrix"
        cmap = "coolwarm"
        normalize = False
        fig, ax = plt.subplots()
        fig.set_size_inches(34, 34)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig("cm.png")
        """


if __name__ == "__main__":
    """
    Example execution:
    python test_plotter.py --testdata_path "P:/casp7/casp7/testing" --model_path "P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16" --result_dir test_results --category 5
    """

    parser = argparse.ArgumentParser(description='TUM Alphafold!')
    parser.add_argument("--testdata_path", help="Path to test set e.g. /path/to/test")
    parser.add_argument("--model_path", help="Path to model e.g. /path/to/model")
    parser.add_argument("--result_dir", help="Path to test set e.g. /path/to/store/results")
    parser.add_argument("--category", help="1:TBM, 2:FM, 3:TBM-Hard, 4:TBM/TBM-Hard, 5:All")

    args = parser.parse_args()
    testdata_path = args.testdata_path
    model_path = args.model_path
    result_dir = args.result_dir
    category = args.category

    plotter(testdata_path, category, model_path, result_dir)
