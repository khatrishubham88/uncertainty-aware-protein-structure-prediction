import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from utils import accuracy_metric, precision_metric
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm
import sys
from network import ResNet

from utils import *


params = {
"crop_size":64, # this is the LxL
"features":"pri-evo", # this will decide the number of channel, with primary 20, pri-evo 41
"padding_value":0, # value to use for padding the sequences, mask is padded by 0 only
"minimum_bin_val":2, # starting bin size
"maximum_bin_val":22, # largest bin size
"num_bins":64,         # num of bins to use
"batch_size":16
}

def create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask, stride):
    batches = []
    for x in range(0,padded_primary.shape[0],stride):
        for y in range(0,padded_primary.shape[0],stride):
            primary_2D_crop = padded_primary[x:x+stride, y:y+stride, :]
            pssm_crop = padded_evol[x:x+stride, y:y+stride, :]
            pri_evol_crop = tf.concat([primary_2D_crop, pssm_crop],axis=2)
            tertiary_crop = padded_dist_map[x:x+stride, y:y+stride]
            tertiary_crop = to_distogram(tertiary_crop, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
            mask_crop = padded_mask[x:x+stride, y:y+stride]

            batches.append((pri_evol_crop, tertiary_crop, mask_crop))

    return batches


def main():
    archi_style = "one_group"
    # archi_style = "two_group_prospr"
    # archi_style = "two_group_alphafold"
    if archi_style=="one_group":
        print("Testing on Minifold architecture!")
    elif archi_style == "two_group_prospr":
        print("Testing on ProSPR architecture!")
    elif archi_style == "two_group_alphafold":
        print("Testing on Alphafold architecture!")
    else:
        print("It is a wrong architecture!")
    if params["features"] == "primary":
        inp_channel = 20
    elif params["features"] == "pri-evo":
        inp_channel = 41

    if archi_style == "one_group":
        num_blocks = [28]
        num_channels = [64]
    elif archi_style == "two_group_prospr":
        num_blocks = [28, 192]
        num_channels = [128, 64]
    elif archi_style == "two_group_alphafold":
        num_blocks = [28, 192]
        num_channels = [256, 128]
    else:
        raise ValueError("Wrong Architecture Selected!")

    nn = ResNet(input_channels=inp_channel, output_channels=64, num_blocks=num_blocks, num_channels=num_channels,
                dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"],
                dropout_rate=0.15, reg_strength=1e-4)
    model = nn.model()

    path = "/home/ghalia/Documents/LabCourse/casp7/testing/1"
    model.load_weights("/home/ghalia/Documents/LabCourse/pcss20-proteinfolding/minifold_trained/custom_model_weights_epochs_30_batch_size_16").expect_partial()
    X = []
    y = []
    mask = []
    for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(path, 'TBM'):
        if (primary != None):
            primary_2D = widen_seq(primary)
            pssm = widen_pssm(evolutionary)
            dist_map = calc_pairwise_distances(tertiary)
            padding_size = math.ceil(primary.shape[0]/params["crop_size"])*params["crop_size"] - primary.shape[0]
            padded_primary = pad_feature2(primary_2D, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_evol = pad_feature2(pssm, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_dist_map = pad_feature2(dist_map, params["crop_size"], params["padding_value"], padding_size, 2)
            padded_mask = pad_feature2(ter_mask, params["crop_size"], params["padding_value"], padding_size, 2)
            batches = create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask, params["crop_size"])
            for batch in batches:
                X.append(batch[0].numpy())
                y.append(batch[1])
                mask.append(batch[2].numpy())

    """
    Begin model evaluation
    """
    X = np.array(X)
    print(X.shape)
    print(X.dtype)
    y_predict = model.predict(X, verbose=1, batch_size=params["batch_size"])
    #pred_distancemap = output_to_distancemaps(y_predict, params["minimum_bin_val"], params["maximum_bin_val"], params["num_bins"])
    print(y_predict.shape)





if __name__=="__main__":
    main()
