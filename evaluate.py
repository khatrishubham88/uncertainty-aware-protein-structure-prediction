import argparse
import glob
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from evaluate_with_MCD import mc_evaluate
from evaluate_with_ts import ts_evaluate
from network_sparse import ResNet, ResNetV2
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm, parse_dataset
from utils import *


from utils_temperature_scaling import *

np.set_printoptions(threshold=np.inf)

sys.setrecursionlimit(10000)

params = {
    "crop_size": 64,  # this will be the  LxL window size
    "features": "pri-evo",  # this will decide the number of channel, with primary 20, pri-evo 41
    "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val": 2,  # starting bin size
    "maximum_bin_val": 22,  # largest bin size
    "num_bins": 64,  # num of bins to use
    "batch_size": 8,  # batch size for training and evaluation
    "modelling_group": 5  # 1: TBM, 2: FM, 3:TBM-hard, 4:TBM/TBM-hard, 5: all
}


def create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask, crop_size, stride):
    batches = []
    for x in range(0, padded_primary.shape[0] - crop_size, stride):
        for y in range(0, padded_primary.shape[0] - crop_size, stride):
            primary_2D_crop = padded_primary[x:x + crop_size, y:y + crop_size, :]
            pssm_crop = padded_evol[x:x + crop_size, y:y + crop_size, :]
            pri_evol_crop = tf.concat([primary_2D_crop, pssm_crop], axis=2)
            tertiary_crop = padded_dist_map[x:x + crop_size, y:y + crop_size]
            tertiary_crop = to_distogram(tertiary_crop, params["minimum_bin_val"], params["maximum_bin_val"],
                                         params["num_bins"])
            mask_crop = padded_mask[x:x + crop_size, y:y + crop_size]
            batches.append((pri_evol_crop, tertiary_crop, mask_crop))

    return batches


def evaluate(testdata_path, model_path, category):
    # path = "/home/ghalia/Documents/alphafold/pcss20-proteinfolding/minifold_trained/custom_model_weights_epochs_30_batch_size_16"
    # path = glob.glob("/home/ghalia/Documents/alphafold/casp7/training/50/*")

    testdata_path = glob.glob(testdata_path + '/*')
    params["modelling_group"] = int(category)
    print('Setting model architecture...')

    ### String manipulations to extract model architecture from model name

    # inp_channel = int(trained_model_name.split('_')[3])
    #
    # num_blocks = (trained_model_name.split('_')[1])[1:-1]
    # if(',' in  num_blocks):
    #     num_blocks = [int(nr) for nr in num_blocks.split(',')]
    # else:
    #     num_blocks = [int(num_blocks)]
    # num_channels = (trained_model_name.split('_')[2])[1:-1]
    # if(',' in  num_channels):
    #     num_channels = [int(nr) for nr in num_channels.split(',')]
    # else:
    #     num_channels = [int(num_channels)]

    model = ResNetV2(input_channels=41, output_channels=params["num_bins"], num_blocks=[28], num_channels=[64],
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     non_linearity='elu', dropout_rate=0.1, reg_strength=1e-4, logits=True, sparse=False,
                     kernel_initializer="he_normal", kernel_regularizer="l2", mc_dropout=False)

    model.load_weights(model_path).expect_partial()
    print('Starting to extract samples from test set...')

    X = []
    y = []
    mask = []
    for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path, params["modelling_group"]):
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

    """
    Begin model evaluation
    """
    X = tf.convert_to_tensor(X)
    y = np.asarray(y)
    mask = tf.convert_to_tensor(mask)
    mask = np.asarray(mask)

    if X.shape[0] % params['batch_size'] != 0:
        drop_samples = X.shape[0] - ((X.shape[0] // params['batch_size']) * params['batch_size'])
        X = X[0:X.shape[0] - drop_samples, :, :]
        mask = mask[0:mask.shape[0] - drop_samples, :, :]
        y = y[0:y.shape[0] - drop_samples, :, :, :]
    y_predict = model.predict(X, verbose=1, batch_size=params["batch_size"])

    samples_acc, total_acc = accuracy_metric(y, y_predict, mask)
    samples_precision, total_precesion = precision_metric(y, y_predict, mask)
    samples_recall , total_recall = recall_metric(y, y_predict, mask)
    f1 = f_beta_score(total_precesion, total_recall, 1)
    print('Contact map based Accuracy: ', total_acc)
    print('Contact map based Precision: ', total_precesion)
    print('Contact map based Recall: ', total_recall)
    print('Contact map based F1_Score: ', f1)

    accuracy, precision, recall, f1, cm = distogram_metrics(y, y_predict, mask, params['minimum_bin_val'],
                                                            params['maximum_bin_val'], params['num_bins'])
    print('Distogram based Accuracy:', accuracy)
    print('Distogram based Precision:', precision)
    print('Distogram based Recall:', recall)
    print('Distogram based F1-score:', f1)

    entropy = entropy_func(y_predict)
    print('Prediction Entropy:', entropy)
    
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
        python evaluate.py --testdata_path "P:/casp7/casp7/testing" --model_path "P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16" --category 5

        python evaluate.py --testdata_path "P:/casp7/casp7/testing" --model_path "P:/proteinfolding_alphafold/clipped_weights_epoch24/chkpnt" --category 2 --ts --temperature_path "P:/proteinfolding_alphafold/temperatures/temperatures_clipped.npy"
    """

    parser = argparse.ArgumentParser(description='TUM Alphafold!')
    parser.add_argument("--testdata_path", help="Path to test set e.g. /path/to/test")
    parser.add_argument("--model_path", help="Path to model e.g. /path/to/model")
    parser.add_argument("--category", help="1:TBM, 2:FM, 3:TBM-Hard, 4:TBM/TBM-Hard, 5:All")
    parser.add_argument("--mc", help="Whether to use the MC Dropout for evaluation", action='store_true')
    parser.add_argument("--sampling", help="Number of sampling to do for MC dropout")
    parser.add_argument("--ts", help="Whether to use the Temperature Scaling for evaluation", action='store_true')
    parser.add_argument("--temperature_path", help="Path to test set e.g. /path/to/temperature.npy")
    parser.add_argument("--plot", help="Whether to plot evaluation set", action='store_true')

    args = parser.parse_args()
    testdata_path = args.testdata_path
    model_path = args.model_path
    category = args.category
    sampling = args.sampling
    temperature_path = args.temperature_path

    if args.mc:
        mc_evaluate(testdata_path, model_path, category, sampling)
    elif args.ts:
        ts_evaluate(testdata_path, model_path, temperature_path, category)
    else:
        evaluate(testdata_path, model_path, category)
