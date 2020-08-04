import glob
import sys

import numpy as np

from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm
from utils import *
from utils_temperature_scaling import *

np.set_printoptions(threshold=np.inf)

sys.setrecursionlimit(10000)

input_channels = 41
num_blocks = [28]
num_channels = [64]
dilation = [1, 2, 4, 8]
dropout_rate = 0.1
reg_strength = 1e-4
kernel_init = "he_normal"

params = {
    "crop_size": 64,  # this will be the  LxL window size
    "features": "pri-evo",  # this will decide the number of channel, with primary 20, pri-evo 41
    "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val": 2,  # starting bin size
    "maximum_bin_val": 22,  # largest bin size
    "num_bins": 64,  # num of bins to use
    "batch_size": 16,  # batch size for training and evaluation
    "modelling_group": 5  # 1: TBM, 2: FM, 3:TBM-hard, 4:TBM/TBM-hard, 5: all
}


def ts_evaluate(testdata_path, model_path, temperature_path, category):
    """Evaluates a model before and after Temperature Scaling for a certain
       category in a test set.
          Args:
            testdata_path: Path to folder containing test data.
            model_path: Path to model weights.
            temperature_path: Path to Numpy binary containing learned temperature.
            category: 1. TBM, 2. FM, 3. TBM-Hard, 4. TBM/TBM-Hard, 5. All
        """
    testdata_path = glob.glob(testdata_path + '/*')
    params["modelling_group"] = int(category)

    print('Start data extraction..')
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

    print("Number of samples in the test data: " + str(len(X)))

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

    # Calculate ECE (Expected Calibration Error) prior to Temperature Scaling.
    model = model_with_logits_output(inp_channel=input_channels, output_channels=params["num_bins"],
                                     num_blocks=num_blocks, num_channels=num_channels, dilation=dilation,
                                     batch_size=params["batch_size"], crop_size=params["crop_size"],
                                     dropout_rate=dropout_rate, reg_strength=reg_strength,
                                     kernel_initializer=kernel_init)
    model.load_weights(model_path).expect_partial()

    temperature = np.load(temperature_path)
    print("Applied Temperature: " + str(temperature))

    y_pred = model.predict(X, verbose=1, batch_size=params["batch_size"])
    accuracy, precision, recall, f1, cm = distogram_metrics(y, softmax(y_pred, axis=-1), mask,
                                                            params['minimum_bin_val'], params['maximum_bin_val'],
                                                            params['num_bins'])
    print('Distogram based Accuracy before TS:', accuracy)
    print('Distogram based Precision before TS:', precision)
    print('Distogram based Recall before TS:', recall)
    print('Distogram based F1-score before TS:', f1)

    ground_truth = y
    mask_original = mask
    max_confidence, prediction, mask_original, true_labels = preprocess_input_ece(y_pred, ground_truth, mask_original,
                                                                                  axis=3)
    accuracies, confidences, bin_lengths = get_bin_info(max_confidence, prediction, true_labels, mask_original,
                                                        bin_size=0.1)

    print("Accuracies before TS: " + str(accuracies))
    print("Confidences before TS: " + str(confidences))

    gaps = np.abs(np.array(accuracies) - np.array(confidences))
    ece = expected_calibration_error(max_confidence, prediction, true_labels, mask_original)

    print("Gaps before TS: " + str(gaps))
    print("ECE before TS: " + str(ece))

    y_pred_sm = softmax(y_pred, axis=-1)
    entropy = entropy_func(y_pred_sm)
    print('Prediction Entropy before TS:', entropy)

    # Uncomment, if temperature should be learned on the test set.
    """
    labels = np.reshape(np.argmax(y, axis=3), (-1))
    y_pred_train = np.reshape(y_pred, (-1, 64))
    mask_train = np.reshape(mask, (-1))

    _, temperature = calibrate_temperature(y_pred_train, labels, mask_train, epochs=100)
    print(temperature)
    """

    y_pred_scaled = predict_with_temperature(y_pred, temperature, training=False)
    accuracy_ts, precision_ts, recall_ts, f1_ts, cm_ts = distogram_metrics(y, softmax(y_pred_scaled, axis=-1), mask,
                                                                           params['minimum_bin_val'],
                                                                           params['maximum_bin_val'],
                                                                           params['num_bins'])
    print('Distogram based Accuracy after TS:', accuracy_ts)
    print('Distogram based Precision after TS:', precision_ts)
    print('Distogram based Recall after TS:', recall_ts)
    print('Distogram based F1-score after TS:', f1_ts)
    max_confidence_scaled, prediction_scaled, mask, true_labels = preprocess_input_ece(y_pred_scaled, y, mask, axis=3)
    accuracies_scaled, confidences_scaled, bin_lengths_scaled = get_bin_info(max_confidence_scaled, prediction_scaled,
                                                                             true_labels, mask, bin_size=0.1)

    print("Accuracies after TS: " + str(accuracies_scaled))
    print("Confidences after TS: " + str(confidences_scaled))
    gaps_scaled = np.abs(np.array(accuracies_scaled) - np.array(confidences_scaled))
    ece_scaled = expected_calibration_error(max_confidence_scaled, prediction_scaled, true_labels, mask)

    print("Gaps after TS: " + str(gaps_scaled))
    print("ECE after TS: " + str(ece_scaled))

    y_pred_scaled_sm = softmax(y_pred_scaled, axis=-1)
    entropy_scaled = entropy_func(y_pred_scaled_sm)
    print('Prediction Entropy after TS:', entropy_scaled)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(22.5, 4), sharex='col', sharey='row')
    rel_diagram_sub(accuracies, confidences, ax[0], name="Before TS")
    rel_diagram_sub(accuracies_scaled, confidences_scaled, ax[1], name="After TS")
    plt.show()
