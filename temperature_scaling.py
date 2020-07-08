import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from dataprovider import DataGenerator
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from tensorflow.keras.optimizers import Adam
from utils_temperature_scaling import *


class TemperatureScaling():
    def __init__(self, temp=1.0, maxiter=1, solver="BFGS"):
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver

    def _loss_fun(self, x, probs, true, mask):
        classes = np.linspace(0, 64, 64, dtype=int)
        scaled_probs = self.predict(probs, x, training=True)
        loss = log_loss(y_true=true, y_pred=scaled_probs, labels=classes, sample_weight=mask)

        return loss

    def fit(self, logits, true, mask):
        #true = true.flatten()  # Flatten y_val
        opt = minimize(self._loss_fun, x0=self.temp, args=(logits, true, mask), options={'maxiter': self.maxiter},
                       method=self.solver)
        self.temp = opt.x[0]

        return opt

    def predict(self, logits, temp=None, training=True):
        axis = 1 if training else 3
        if not temp:
            return softmax(logits / self.temp, axis=axis)
        else:
            return softmax(logits / temp, axis=axis)


def fit_TemperatureCalibration(X_val, y_val, mask, T=tf.Variable(tf.ones(shape=(1, ))), epochs=100):
    history = []
    optimizer = Adam(learning_rate=1e-3)

    def cost(T, x, y, mask):
        for i in range(0, x.shape[1]):
            x[:, i] = tf.multiply(x=x[:, i], y=mask)
        y = tf.multiply(x=y, y=mask)
        scaled_logits = tf.multiply(x=x, y=1.0/T)

        cost_value = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scaled_logits, labels=y)
        )

        return cost_value

    def grad(T, x, y):
        with tf.GradientTape() as tape:
            cost_value = cost(T, x, y, mask)

        return cost_value, tape.gradient(cost_value, T)

    for epoch in range(epochs):
        train_cost, grads = grad(T, X_val, y_val)
        optimizer.apply_gradients(zip([grads], [T]))
        history.append([train_cost, T.numpy()[0]])

    history = np.asarray(history)
    temperature = history[-1, -1]

    return history, temperature


if __name__ == '__main__':
    train_path = glob.glob("P:/casp7/casp7/training/100/*")
    val_path = glob.glob("P:/casp7/casp7/validation/1")
    train_plot = True
    validation_plot = True

    params = {
        "crop_size": 64,  # this is the LxL
        "datasize": None,
        "features": "pri-evo",  # this will decide the number of channel, with primary 20, pri-evo 41
        "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
        "minimum_bin_val": 2,  # starting bin size
        "maximum_bin_val": 22,  # largest bin size
        "num_bins": 64,  # num of bins to use
        "batch_size": 16,  # batch size for training, check if this is needed here or should be done directly in fit?
        "shuffle": True,  # if wanna shuffle the data, this is not necessary
        "shuffle_buffer_size": None,  # if shuffle is on size of shuffle buffer, if None then =batch_size
        "random_crop": True,  # if cropping should be random, this has to be implemented later
        "flattening": True,
        # "take":8,
        "epochs": 30,
        "prefetch": True,
        "val_path": val_path,
        "validation_thinning_threshold": 50,
        "training_validation_ratio": 0.2,
        # "experimental_val_take": 2
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

    dataprovider = DataGenerator(val_path, **params)
    if params.get("train_path", None) is not None:
        validation_data = dataprovider.get_validation_dataset()
        if params.get("experimental_val_take", None) is not None:
            validation_steps = params.get("experimental_val_take", None)
        else:
            validation_steps = dataprovider.get_validation_length()

    features = 'pri-evo'
    output_channels = 64
    num_blocks = [28]
    num_channels = [64]
    dilation = [1, 2, 4, 8]
    batch_size = 16
    crop_size = 64
    dropout_rate = 0.1
    reg_strength = 1e-4
    weights_path = "P:/proteinfolding_alphafold/minifold_trained/custom_model_weights_epochs_30_batch_size_16"

    model = model_with_logits_output(features=features, model_weights=weights_path, output_channels=output_channels,
                                     num_blocks=num_blocks, num_channels=num_channels, dilation=dilation,
                                     batch_size=batch_size, crop_size=crop_size, dropout_rate=dropout_rate,
                                     reg_strength=reg_strength)
    temperatures = [1.]
    ece_before = []
    ece_after = []
    if validation_plot:
        if params.get("val_path", None) is not None:
            tp = TemperatureScaling()
            for j, val in enumerate(validation_data):
                X, ground_truth, mask = val
                mask = mask.numpy()
                ground_truth = ground_truth.numpy()
                output = model.predict(X)
                confids = softmax(output, axis=3)
                labels = np.argmax(ground_truth, axis=3)
                preds = np.argmax(confids, axis=3)
                confs = np.max(confids, axis=3)

                confs = np.reshape(confs, -1)
                preds = np.reshape(preds, -1)
                mask = np.reshape(mask, -1)
                labels = np.reshape(labels, -1)

                accuracies, confidences, bin_lengths = get_bin_info(confs, preds, labels, mask, bin_size=0.1)
                ece = expected_calibration_error(confs, preds, labels, mask)
                ece_before.append(ece)
                print(str(j) + ". ECE: " + str(ece))
                """
                print("Before Temperature Scaling:")
                print("Accuracies: " + str(accuracies))
                print("Confidences: " + str(confidences))
                print("ECE: " + str(ece))
                print("================================")

                
                X, ground_truth, mask = val
                mask = mask.numpy()
                ground_truth = ground_truth.numpy()
                output = model.predict(X)
                output = np.reshape(output, (-1, 64))
                mask = np.reshape(mask, (-1))
                opt = tp.fit(output, labels, mask)
                print("Temperature: " + str(tp.temp))
                """

                X, ground_truth, mask = val
                mask = mask.numpy()
                ground_truth = ground_truth.numpy()
                output = model.predict(X)
                output = np.reshape(output, (-1, 64))
                mask = np.reshape(mask, (-1))
                if j % 5 != 0:
                    if j == 0:
                        history, T = fit_TemperatureCalibration(output, labels, mask)
                    else:
                        history, T = fit_TemperatureCalibration(output, labels, mask, T=tf.Variable([1.]))
                    print(str(j) + ". Temperature: " + str(T))
                    temperatures.append(T)
                else:
                    print(str(j) + ". Predicting with learned temperature..")

                X, ground_truth, mask = val
                mask = mask.numpy()
                ground_truth = ground_truth.numpy()
                logits = model.predict(X)
                print(str(j) + ". Temperature: " + str(sum(temperatures)/len(temperatures)))
                confids_scaled = tp.predict(logits, temp=sum(temperatures)/len(temperatures), training=False)
                labels_scaled = np.argmax(ground_truth, axis=3)
                preds_scaled = np.argmax(confids_scaled, axis=3)
                confs_scaled = np.max(confids_scaled, axis=3)

                confs_scaled = np.reshape(confs_scaled, -1)
                preds_scaled = np.reshape(preds_scaled, -1)
                mask = np.reshape(mask, -1)
                labels_scaled = np.reshape(labels_scaled, -1)

                accuracies_scaled, confidences_scaled, bin_lengths_scaled = get_bin_info(confs_scaled, preds_scaled,
                                                                                         labels_scaled, mask,
                                                                                         bin_size=0.1)

                ece_scaled = expected_calibration_error(confs_scaled, preds_scaled, labels_scaled, mask)
                ece_after.append(ece_scaled)
                print(str(j) + ". ECE: " + str(ece_scaled))
                print("================================")

                """
                print("After Temperature Scaling:")
                print("Accuracies: " + str(accuracies_scaled))
                print("Confidences: " + str(confidences_scaled))
                print("ECE: " + str(ece_scaled))
                print("================================")
                
                plt.style.use('ggplot')
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22.5, 4), sharex='col', sharey='row')
                rel_diagram_sub(accuracies, confidences, ax)
                plt.show()

                plt.style.use('ggplot')
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22.5, 4), sharex='col', sharey='row')
                rel_diagram_sub(accuracies_scaled, confidences_scaled, ax)
                plt.show()
                """

    print(sum(ece_before)/len(ece_before))
    print(sum(temperatures)/len(temperatures))
    print(sum(ece_after)/len(ece_after))