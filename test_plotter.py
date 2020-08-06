from network import ResNetV2
<<<<<<< HEAD
=======
from dataprovider import DataGenerator
>>>>>>> 4584528ae49dc4a90fc499fdf3a3dbcae92fbeb7
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
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

def plotter(output, y, mask, params, result_dir):
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
    print('Begin Plotting...')
    
    y_predict = output_to_distancemaps(output, 2, 22, 64)
    gt = output_to_distancemaps(y, 2, 22, 64)

    accuracy, precision, recall, f1, cm = distogram_metrics(y, output, mask, params['minimum_bin_val'],
                                                           params['maximum_bin_val'], params['num_bins'])
    
    contact_maps_pred = contact_map_from_distancemap(y_predict)
    contact_maps_true = contact_map_from_distancemap(gt)
    sample_accuracies, _ = accuracy_metric(y, output, mask)
    sample_precisions, cm_precision = precision_metric(y, output, mask)
    sample_recalls, cm_recall = recall_metric(y, output, mask)
    _ = f_beta_score(cm_precision, cm_recall, beta=1)

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
    fig.savefig(result_dir + "/cm.png")

