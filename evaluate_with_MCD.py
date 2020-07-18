from network_sparse import ResNet, ResNetV2
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm, parse_dataset
import sys
import glob
from utils import *
from scipy.stats import entropy
np.set_printoptions(threshold=np.inf)

sys.setrecursionlimit(10000)

params = {
"crop_size":64,             # this will be the  LxL window size
"features":"pri-evo",       # this will decide the number of channel, with primary 20, pri-evo 41
"padding_value":0,          # value to use for padding the sequences, mask is padded by 0 only
"minimum_bin_val":2,        # starting bin size
"maximum_bin_val":22,       # largest bin size
"num_bins":64,              # num of bins to use
"batch_size":6,             # batch size for training and evaluation
"modelling_group":5         # 1: TBM, 2: FM, 3:TBM-hard, 4:TBM/TBM-hard, 5: all
}

def entropy_func(y_predict):
    sample_entropy = np.zeros((y_predict.shape[1], y_predict.shape[2]))
    samples_entropy = []
    tot_entropy = 0
    for sample in range(y_predict.shape[0]):
        for x in range(y_predict[sample].shape[0]):
            for y in range(y_predict[sample].shape[1]):
                ent = entropy(y_predict[sample][x,y])
                sample_entropy[x,y] = ent

        sample_mean = np.mean(sample_entropy)
        samples_entropy.append(sample_mean)
    return np.mean(sample_entropy)

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


def mc_evaluate(testdata_path, model_path, category, sampling):
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
                dropout_rate=0.15, logits=True, mc_dropout=True, mc_sampling=int(sampling))

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
            batches = create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask, params["crop_size"])
            for batch in batches:
                X.append(batch[0])            # batch[0] of type eager tensor
                y.append(batch[1])            # batch[1] of type ndarray
                mask.append(batch[2])         #batch[2] of type eager tensor

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
    if( X.shape[0] % params['batch_size'] != 0 ):
        drop_samples = X.shape[0] - ((X.shape[0] // params['batch_size']) * params['batch_size'])
        X = X[0:X.shape[0]-drop_samples,:,:]
        mask = mask[0:mask.shape[0]-drop_samples,:,:]
        y = y[0:y.shape[0]-drop_samples,:,:,:]
    
    mc_pred, mean_predict = model.mc_predict(X)
    print(mc_pred.shape)
    print(mean_predict.shape)
    print(mean_predict[0, 0, 0, :])
    entropy = entropy_func(mean_predict)
    print('Prediction Entropy with MC:', entropy)
    #for iter in range(mc_pred.shape[0]):
