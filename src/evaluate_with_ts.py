import glob
import sys

import numpy as np
from test_plotter import plotter
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



def ts_evaluate(X, y, mask, model_path, temperature_path, params, plot=False, result_dir=None):
    """Evaluates a model before and after Temperature Scaling for a certain
    category in a test set.
      Args:
        X: Input data as TensorFlow tensor.
        y: Ground truth as Numpy array.
        mask: Masking tensor as Numpy array.
        model_path: String containing path to model weights.
        temperature_path: Path to Numpy binary containing learned temperature.
    """

    print('Begin model evaluation...')

    """
    Begin model evaluation
    """
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
    
    if plot:
        plotter(y_pred_scaled_sm, y, mask, params, result_dir)
