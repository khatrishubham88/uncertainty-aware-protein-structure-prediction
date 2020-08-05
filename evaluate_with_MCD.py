import glob
import sys

import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

from network import ResNetV2
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm, prepare_test_set
from utils import *
from mc_utils import *
from test_plotter import plotter

np.set_printoptions(threshold=np.inf)
sys.setrecursionlimit(10000)

def mc_evaluate(X, y, mask, model_path, params, sampling, plot=False, result_dir=None):
    # testdata_path = glob.glob(testdata_path + '/*')
    # params["modelling_group"] = int(category)
    print('Setting model architecture...')

    model = ResNetV2(input_channels=41, output_channels=params["num_bins"], num_blocks=[28], num_channels=[64],
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     non_linearity='elu', dropout_rate=0.1, reg_strength=1e-4, logits=True, sparse=False,
                     kernel_initializer="he_normal", kernel_regularizer="l2", mc_dropout=True, mc_sampling=int(sampling))

    model.load_weights(model_path).expect_partial()
    print('Begin model evaluation...')
    """
    Begin model evaluation
    """
    _, mean_predict, mis_spec = model.mc_predict(X)
    
    del model
    
    # model noise computation
    model = ResNetV2(input_channels=41, output_channels=params["num_bins"], num_blocks=[28], num_channels=[64],
                     dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                     non_linearity='elu', dropout_rate=0.1, reg_strength=1e-4, logits=True, sparse=False,
                     kernel_initializer="he_normal", kernel_regularizer="l2", mc_dropout=False, mc_sampling=int(sampling))

    model.load_weights(model_path).expect_partial()
    
    y_predict = model.predict(X, verbose=1, batch_size=params["batch_size"])
    model_noise = total_model_noise(y, y_predict, params["num_bins"])
    
    # print("model_noise = {}, mis_spec = {}, type_mis_spec = {}, type_model_noise = {}".format(model_noise, mis_spec, type(mis_spec), type(model_noise)))
    model_uq = math.sqrt(model_noise**2 + mis_spec**2)
    accuracy, precision, recall, f1, cm = distogram_metrics(y, mean_predict, mask, params['minimum_bin_val'],
                                                            params['maximum_bin_val'], params['num_bins'])
    
    print('Distogram based Accuracy:', accuracy)
    print('Distogram based Precision:', precision)
    print('Distogram based Recall:', recall)
    print('Distogram based F1-score:', f1)

    entropy = entropy_func(mean_predict)


    print('Prediction Entropy with MC:', entropy)
    print("Model Prediction Uncertainity: {}".format(model_uq))
    _, cm_accuracy = accuracy_metric(y, mean_predict, mask)
    _, cm_precision = precision_metric(y, mean_predict, mask)
    _, cm_recall = recall_metric(y, mean_predict, mask)
    cm_fscore = f_beta_score(cm_precision, cm_recall, beta=1)

    print("Contact Map Accuracy: " + str(cm_accuracy))
    print("Contact Map Precision: " + str(cm_precision))
    print("Contact Map Recall: " + str(cm_recall))
    print("Contact Map F1-Score: " + str(cm_fscore))
    if plot:
        plotter(mean_predict, y, mask, params, result_dir)