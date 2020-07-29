import numpy as np
import tensorflow as tf

from network_sparse import ResNetV2
from tensorflow.keras.optimizers import Adam


def softmax(x, axis):
    exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exps / np.sum(exps, axis=axis, keepdims=True)


def model_with_logits_output(inp_channel=41, output_channels=64, num_blocks=None, num_channels=None,
                             dilation=None, batch_size=16, crop_size=64, dropout_rate=0.1, reg_strength=1e-4,
                             kernel_initializer="he_normal"):

    if num_channels is None:
        num_channels = [64]
    if num_blocks is None:
        num_blocks = [28]
    if dilation is None:
        dilation = [1, 2, 4, 8]

    model = ResNetV2(input_channels=inp_channel, output_channels=output_channels, num_blocks=num_blocks,
                     num_channels=num_channels, dilation=dilation, batch_size=batch_size, crop_size=crop_size,
                     non_linearity='elu', dropout_rate=dropout_rate, reg_strength=reg_strength, logits=False,
                     kernel_initializer=kernel_initializer, kernel_regularizer="l2")

    return model


def preprocess_input_ece(y_pred, y_true, mask, axis=3):
    confids = softmax(y_pred, axis=axis)
    labels = np.argmax(y_true, axis=axis)
    preds = np.argmax(confids, axis=axis)
    confs = np.max(confids, axis=axis)

    max_confidence = np.reshape(confs, -1)
    prediction = np.reshape(preds, -1)
    mask = np.reshape(mask, -1)
    true_labels = np.reshape(labels, -1)

    return max_confidence, prediction, mask, true_labels


def get_bin_info(conf, pred, true, mask, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_accuracy_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true, mask)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths


def compute_accuracy_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true, mask):
    filtered_tuples = [x for x in zip(pred, true, conf, mask) if
                       conf_thresh_lower < x[2] <= conf_thresh_upper and x[3] == 1]

    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


def rel_diagram_sub(accs, confs, ax, n_bins=10, name="Reliability Diagram", xname="", yname=""):
    acc_conf = np.column_stack([accs, confs])
    acc_conf.sort(axis=1)
    outputs = acc_conf[:, 0]
    gap = acc_conf[:, 1]

    bin_size = 1 / n_bins
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)

    # Plot gap first, so its below everything
    gap_plt = ax.bar(positions, gap, width=bin_size, edgecolor="red", color="red", alpha=0.3, label="Gap", linewidth=2,
                     zorder=2)

    # Next add error lines
    # for i in range(n_bins):
    #     plt.plot([i/n_bins, 1], [0, (n_bins - i) / n_bins], color="red", alpha=0.5, zorder=1)

    # Bars with outputs
    output_plt = ax.bar(positions, outputs, width=bin_size, edgecolor="black", color="blue", label="Outputs", zorder=3)

    # Line plot with center line.
    ax.set_aspect('equal')
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.legend(handles=[gap_plt, output_plt])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(name, fontsize=24)
    ax.set_xlabel(xname, fontsize=22, color="black")
    ax.set_ylabel(yname, fontsize=22, color="black")


def expected_calibration_error(conf, pred, true, mask, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = np.count_nonzero(mask)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_accuracy_bin(conf_thresh - bin_size, conf_thresh, conf, pred, true, mask)
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weighted difference to ECE

    return ece


def predict_with_temperature(logits, temp=None, training=True):
    shape = logits.shape
    logits = np.reshape(logits, (-1, 64))
    logits = logits / temp
    if training:
        out = softmax(np.reshape(logits, shape), axis=-1)
    else:
        out = np.reshape(logits, shape)

    return out


def calibrate_temperature(x_val, y_val, mask, temp=tf.Variable(tf.ones(shape=(1,))), epochs=3, lr=1e-3):
    history = []
    optimizer = Adam(learning_rate=lr)

    def cost(temp, x, y, mask):
        for i in range(0, x.shape[1]):
            x[:, i] = tf.multiply(x=x[:, i], y=mask)
        y = tf.multiply(x=y, y=mask)
        scaled_logits = tf.multiply(x=x, y=1.0/np.array(temp))

        cost_value = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scaled_logits, labels=y)
        )

        return cost_value

    def grad(temp, x, y):
        with tf.GradientTape() as tape:
            cost_value = cost(temp, x, y, mask)

        return cost_value, tape.gradient(cost_value, temp)

    for epoch in range(epochs):
        train_cost, grads = grad(temp, x_val, y_val)
        optimizer.apply_gradients(zip([grads], [temp]))
        history.append([train_cost, temp.numpy()])

    history = np.asarray(history)
    temperature = history[-1, -1]

    return history, temperature
