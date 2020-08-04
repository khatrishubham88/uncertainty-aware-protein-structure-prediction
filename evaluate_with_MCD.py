import glob
import sys

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

from network import ResNetV2
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm
from utils import *

np.set_printoptions(threshold=np.inf)
sys.setrecursionlimit(10000)

params = {
"crop_size":64,             # this will be the  LxL window size
"features":"pri-evo",       # this will decide the number of channel, with primary 20, pri-evo 41
"padding_value":0,          # value to use for padding the sequences, mask is padded by 0 only
"minimum_bin_val":2,        # starting bin size
"maximum_bin_val":22,       # largest bin size
"num_bins":64,              # num of bins to use
"batch_size":8,             # batch size for training and evaluation
"modelling_group":5         # 1: TBM, 2: FM, 3:TBM-hard, 4:TBM/TBM-hard, 5: all
}


def mc_evaluate(testdata_path, model_path, category, sampling):
    testdata_path = glob.glob(testdata_path + '/*')
    params["modelling_group"] = int(category)
    print('Setting model architecture...')

    model = ResNetV2(input_channels=41, output_channels=params["num_bins"], num_blocks=[28], num_channels=[64],
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     non_linearity='elu', dropout_rate=0.1, reg_strength=1e-4, logits=True, sparse=False,
                     kernel_initializer="he_normal", kernel_regularizer="l2", mc_dropout=True, mc_sampling=int(sampling))

    model.load_weights(model_path).expect_partial()
    print('Starting to extract features from test set...')

    X = []
    y = []
    mask = []
    for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path, params["modelling_group"]):
        if (primary != None):
            primary_2D = widen_seq(primary)
            pssm = widen_pssm(evolutionary)
            dist_map = calc_pairwise_distances(tertiary)
            padding_size = math.ceil(primary.shape[0]/params["crop_size"])*params["crop_size"] - primary.shape[0]
            padded_primary = pad_feature2(primary_2D, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_evol = pad_feature2(pssm, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_dist_map = pad_feature2(dist_map, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_mask = pad_feature2(ter_mask, params["crop_size"], params["padding_value"], padding_size, 2)
            batches = create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask,
                                           params["crop_size"], params["crop_size"], params["minimum_bin_val"],
                                           params["maximum_bin_val"], params["num_bins"])
            for i, batch in enumerate(batches):
                X.append(batch[0])            # batch[0] of type eager tensor
                y.append(batch[1])            # batch[1] of type ndarray
                mask.append(batch[2])         # batch[2] of type eager tensor

    print('Finish Feature Extraction...')
    print('Begin model evaluation...')
    """
    Begin model evaluation
    """
    X = tf.convert_to_tensor(X)
    y = np.asarray(y)
    mask = tf.convert_to_tensor(mask)
    mask = np.asarray(mask)
    print(X.shape)
    if X.shape[0] % params['batch_size'] != 0:
        drop_samples = X.shape[0] - ((X.shape[0] // params['batch_size']) * params['batch_size'])
        X = X[0:X.shape[0] - drop_samples, :, :]
        mask = mask[0:mask.shape[0] - drop_samples, :, :]
        y = y[0:y.shape[0] - drop_samples, :, :, :]

    _, mean_predict = model.mc_predict(X)


    accuracy, precision, recall, f1, cm = distogram_metrics(y, mean_predict, mask, params['minimum_bin_val'],
                                                            params['maximum_bin_val'], params['num_bins'])
    print('Distogram based Accuracy:', accuracy)
    print('Distogram based Precision:', precision)
    print('Distogram based Recall:', recall)
    print('Distogram based F1-score:', f1)

    entropy = entropy_func(mean_predict)


    print('Prediction Entropy with MC:', entropy)
    _, cm_accuracy = accuracy_metric(y, mean_predict, mask)
    _, cm_precision = precision_metric(y, mean_predict, mask)
    _, cm_recall = recall_metric(y, mean_predict, mask)
    cm_fscore = f_beta_score(cm_precision, cm_recall, beta=1)

    print("Contact Map Accuracy: " + str(cm_accuracy))
    print("Contact Map Precision: " + str(cm_precision))
    print("Contact Map Recall: " + str(cm_recall))
    print("Contact Map F1-Score: " + str(cm_fscore))

    ##TODO ECE
