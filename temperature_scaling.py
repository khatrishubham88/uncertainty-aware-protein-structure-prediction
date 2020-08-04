import argparse
import glob
import os

from dataprovider import DataGenerator
from utils_temperature_scaling import *


def learn_temperature(traindata_path, valdata_path, model_path, epochs, iterations_per_batch, lr):
    """Learns the temperature for given validation set and model weights.
          Args:
            traindata_path: Path to folder containing training data.
            valdata_path: Path to validation data.
            model_path: Path to model weights.
            epochs: Number of sweeps through validation set to learn temperature.
            iterations_per_batch: Number of optimization steps on each batch.
        """
    train_path = glob.glob(traindata_path)
    val_path = glob.glob(valdata_path)

    input_channels = 41
    num_blocks = [28]
    num_channels = [64]
    dilation = [1, 2, 4, 8]
    dropout_rate = 0.1
    reg_strength = 0.0001
    kernel_init = "he_normal"
    weights_path = model_path

    params = {
        "crop_size": 64,  # this is the LxL
        "datasize": None,
        "features": "pri-evo",  # this will decide the number of input channels with primary = 20 and pri-evo = 41
        "padding_value": 0,  # value to use for padding the sequences, mask is padded by 0 only
        "minimum_bin_val": 2,  # starting bin size
        "maximum_bin_val": 22,  # largest bin size
        "num_bins": 64,  # num of bins to use
        "batch_size": 16,  # batch size for training, check if this is needed here or should be done directly in fit?
        "shuffle": True,  # if wanna shuffle the data, this is not necessary
        "shuffle_buffer_size": None,  # if shuffle is on size of shuffle buffer, if None then =batch_size
        "random_crop": True,  # if cropping should be random, this has to be implemented later
        "flattening": True,
        "epochs": int(epochs),
        "prefetch": True,
        "val_path": val_path,
        "validation_thinning_threshold": 50,
        "training_validation_ratio": 0.2,
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

    model = model_with_logits_output(inp_channel=input_channels, output_channels=params["num_bins"],
                                     num_blocks=num_blocks, num_channels=num_channels, dilation=dilation,
                                     batch_size=params["batch_size"], crop_size=params["crop_size"],
                                     dropout_rate=dropout_rate, reg_strength=reg_strength,
                                     kernel_initializer=kernel_init)

    model.load_weights(weights_path).expect_partial()
    model.summary()

    temperatures = [1.0]
    ece_before = []
    ece_after = []
    if params.get("val_path", None) is not None:
        for j, val in enumerate(validation_data):
            X, ground_truth, mask = val
            mask = mask.numpy()
            y_true = ground_truth.numpy()
            y_pred = model.predict(X, batch_size=params["batch_size"])
            max_confidence, prediction, mask, true_labels = preprocess_input_ece(y_pred, y_true, mask, axis=3)
            ece = expected_calibration_error(max_confidence, prediction, true_labels, mask)
            ece_before.append(ece)

            X, ground_truth, mask = val
            mask = mask.numpy()
            y_true = ground_truth.numpy()
            y_pred = model.predict(X, batch_size=params["batch_size"])
            y_true = np.reshape(np.argmax(y_true, axis=3), (-1))
            y_pred = np.reshape(y_pred, (-1, 64))
            mask = np.reshape(mask, (-1))

            if j == 0:
                history, T = calibrate_temperature(y_pred, y_true, mask, epochs=int(iterations_per_batch), lr=float(lr))
            else:
                history, T = calibrate_temperature(y_pred, y_true, mask, temp=tf.Variable(temperatures[-1]),
                                                   epochs=int(iterations_per_batch), lr=float(lr))
            temperatures.append(T)

            X, ground_truth, mask = val
            mask = mask.numpy()
            y_true = ground_truth.numpy()
            y_pred = model.predict(X, batch_size=params["batch_size"])

            y_pred = predict_with_temperature(y_pred, temp=temperatures[-1], training=False)
            max_confidence, prediction, mask, true_labels = preprocess_input_ece(y_pred, y_true, mask, axis=3)
            ece_scaled = expected_calibration_error(max_confidence, prediction, true_labels, mask)
            ece_after.append(ece_scaled)

            print("(" + str(j + 1) + "/" + str(dataprovider.get_validation_length()*params["epochs"]) +
                  "): ECE of batch before Temperature Scaling: " + str(ece))
            print("(" + str(j + 1) + "/" + str(dataprovider.get_validation_length()*params["epochs"])
                  + "): Current temperature: " + str(temperatures[-1]))
            print("(" + str(j + 1) + "/" + str(dataprovider.get_validation_length()*params["epochs"])
                  + "): ECE of batch after Temperature Scaling: " + str(ece_scaled))
            print("===========================================================================")

    print("Final Temperature: " + str(temperatures[-1][0]))
    temperatures_dir = "temperatures"
    if os.path.isdir(temperatures_dir) is False:
        os.mkdir(temperatures_dir)
    name_temperature_binary_file = temperatures_dir + "/" + "temperatures_unclipped" # 1.3075706
    np.save(name_temperature_binary_file, temperatures[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temperature Scaling.')
    parser.add_argument("--traindata_path", help="Path to train set e.g. /path/to/train")
    parser.add_argument("--valdata_path", help="Path to validation set e.g. /path/to/val")
    parser.add_argument("--modelweight_path", help="Path to model weights e.g. /path/to/model_weight")
    parser.add_argument("--epochs", help="Number of passes through the validation data.")
    parser.add_argument("--iterations", help="Number of optimization loops per batch.")
    parser.add_argument("--lr", help="Learning rate for optimization.")

    args = parser.parse_args()
    traindata_path = args.traindata_path
    valdata_path = args.valdata_path
    modelweight_path = args.modelweight_path
    epochs = args.epochs
    iterations = args.iterations
    lr = args.lr

    learn_temperature(traindata_path=traindata_path, valdata_path=valdata_path, model_path=modelweight_path,
                      epochs=epochs, iterations_per_batch=iterations, lr=lr)
