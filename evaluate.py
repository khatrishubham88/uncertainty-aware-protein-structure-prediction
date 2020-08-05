import argparse
import glob
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from evaluate_with_MCD import mc_evaluate
from evaluate_with_ts import ts_evaluate
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm
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
    "batch_size": 1,  # batch size for training and evaluation
    "modelling_group": 5  # 1: TBM, 2: FM, 3:TBM-hard, 4:TBM/TBM-hard, 5: all
}


def evaluate(testdata_path, model_path, category):
    testdata_path = glob.glob(testdata_path + '/*')
    params["modelling_group"] = int(category)
    print('Setting model architecture...')

    model_logits = model_with_logits_output(inp_channel=41, output_channels=params["num_bins"],
                                            num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                                            batch_size=params["batch_size"], crop_size=params["crop_size"],
                                            dropout_rate=0.1, reg_strength=1e-4, kernel_initializer="he_normal")

    # model.load_weights(model_path).expect_partial()
    model_logits.load_weights(model_path).expect_partial()
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
                                           params["crop_size"], params["crop_size"], params["minimum_bin_val"],
                                           params["maximum_bin_val"], params["num_bins"])
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

    ground_truth = y
    mask_original = mask
    y_predict_logits = model_logits.predict(X, verbose=1, batch_size=params["batch_size"])
    max_confidence, prediction, mask_original, true_labels = preprocess_input_ece(y_predict_logits, ground_truth,
                                                                                  mask_original, axis=3)
    ece = expected_calibration_error(max_confidence, prediction, true_labels, mask_original)
    y_predict = softmax(y_predict_logits, axis=-1)

    samples_acc, total_acc = accuracy_metric(y, y_predict, mask)
    samples_precision, total_precesion = precision_metric(y, y_predict, mask)
    samples_recall, total_recall = recall_metric(y, y_predict, mask)
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
    print('ECE: ', ece)


if __name__ == "__main__":
    """
    Example execution:
        python evaluate.py --testdata_path "P:/casp7/casp7/testing" --model_path "P:/proteinfolding_alphafold/unweighted_model/custom_model_weights_epochs_30_batch_size_16" --category 5

        python evaluate.py --testdata_path "P:/casp7/casp7/testing" --model_path "P:/proteinfolding_alphafold/clipped_weights_epoch24/chkpnt" --category 2 --ts --temperature_path "P:/proteinfolding_alphafold/temperatures/temperature_weighted.npy"
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
