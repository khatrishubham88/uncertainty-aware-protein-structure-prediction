import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from utils import accuracy_metric, precision_metric, distogram_accuracy_metric, distogram_precision_metric, distogram_recall_metric, f_beta_score
from readData_from_TFRec import parse_test_dataset, widen_seq, widen_pssm
import sys
from network import ResNet
import glob
from utils import *
from scipy.stats import entropy
import argparse
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
    sample_entropy = []
    tot_entropy = 0
    for sample in range(y_predict.shape[0]):
        predictions = np.amax(y_predict[sample], axis=2)
        ent = entropy(predictions.flatten())
        sample_entropy.append(ent)
        tot_entropy = tot_entropy + ent
    return sample_entropy, tot_entropy / y_predict.shape[0]


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


def evaluate(testdata_path, model_path, category):
##"/home/ghalia/Documents/alphafold/pcss20-proteinfolding/minifold_trained/custom_model_weights_epochs_30_batch_size_16"
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

    nn = ResNet(input_channels=41, output_channels=params["num_bins"], num_blocks=[28], num_channels=[64],
                dilation=[1, 2, 4, 8], batch_size=params["batch_size"], crop_size=params["crop_size"])
    model = nn.model()
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
    # i = 0
    # for sample in range(mask.shape[0]):
    #      if(np.all(mask[sample] == 0)):
    #          i = i +1
    #          print(mask[sample])
    #          plt.figure()
    #          plt.title("Mask")
    #          plt.imshow(mask[sample], cmap='viridis_r')
    #          plt.savefig("/home/ghalia/Documents/alphafold/masks/masks-TBM/"+str(i)+".png")
    #          plt.close("all")
    if( X.shape[0] % params['batch_size'] != 0 ):
        drop_samples = X.shape[0] - ((X.shape[0] // params['batch_size']) * params['batch_size'])
        X = X[0:X.shape[0]-drop_samples,:,:]
        mask = mask[0:mask.shape[0]-drop_samples,:,:]
        y = y[0:y.shape[0]-drop_samples,:,:,:]
    y_predict = model.predict(X, verbose=1, batch_size=params["batch_size"])

    # samples_acc, total_acc = accuracy_metric(y, y_predict, mask)
    # samples_precision, total_precesion = precision_metric(y, y_predict, mask)
    # samples_recall , total_recall = recall_metric(y, y_predict, mask)
    # f1 = f_beta_score(total_precesion, total_recall, 1)
    # print('Contact map based Accuracy: ', total_acc)
    # print('Contact map based Precision: ', total_precesion)
    # print('Contact map based Recall: ', total_recall)
    # print('Contact map based F1_Score: ', f1)

    #distogram_acc_samples, distogram_acc_total = distogram_accuracy_metric(y, y_predict, mask, params['minimum_bin_val'],
    #                                                                    params['maximum_bin_val'], params['num_bins'])
    distogram_prec_samples, distogram_prec_total = distogram_precision_metric(y, y_predict, mask, params['minimum_bin_val'],
                                                                        params['maximum_bin_val'], params['num_bins'])
    distogram_recall_samples, distogram_recall_total = distogram_recall_metric(y, y_predict, mask, params['minimum_bin_val'],
                                                                        params['maximum_bin_val'], params['num_bins'])
    distogram_f1_score = f_beta_score(distogram_prec_total, distogram_recall_total, 1)

    #print('Distogram based Accuracy:', distogram_acc_total)
    print('Distogram based Precision:', distogram_prec_total)
    print('Distogram based Recall:', distogram_recall_total)
    print('Distogram based F1-score:', distogram_f1_score)

    samples_entropy, total_entropy =  entropy_func(y_predict)
    print('Prediction Entropy:', total_entropy)


if __name__=="__main__":
    """
    Example execution:
    python evaluate.py --testdata_path '/home/ghalia/Documents/alphafold/casp9/testing'
    --model_path '/home/ghalia/Documents/alphafold/pcss20-proteinfolding/minifold_trained/custom_model_weights_epochs_30_batch_size_16'
     --category 5
    """
    parser = argparse.ArgumentParser(description='TUM Alphafold!')
    parser.add_argument("--testdata_path",help="path to test set e.x /path/to/test")
    parser.add_argument("--model_path", help="path to model e.x /path/to/model")
    parser.add_argument("--category", help="1:TBM, 2:FM, 3:TBM-Hard, 4:TBM/TBM-Hard, 5:all")
    args = parser.parse_args()
    testdata_path = args.testdata_path
    model_path = args.model_path
    category = args.category
    evaluate(testdata_path, model_path, category)
