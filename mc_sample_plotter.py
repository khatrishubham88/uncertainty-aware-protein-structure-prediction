from network_sparse import ResNet, ResNetV2
# from dataprovider_sparse import DataGenerator
from dataprovider import DataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import glob
import os
import time
import warnings
from utils import accuracy_metric, precision_metric
import sys

from utils import *

sys.setrecursionlimit(100000)
plt.style.use("ggplot")
def main():
    result_dir = "test_results"
    val_result_dir = result_dir + "/val_data"
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
        if os.path.isdir(val_result_dir) is False:
            os.mkdir(val_result_dir)
    file_path = "mcd.npy"
    val_accuracy = np.load(file_path)
    val_accuracy = val_accuracy.transpose(1,0,2)
    val_accuracy = val_accuracy.reshape(val_accuracy.shape[0],-1)
    val_accuracy = np.transpose(val_accuracy)
    for i in range(val_accuracy.shape[0]):
        plt.figure()
        plt.title("Accuracy distribution")
        plt.hist(val_accuracy[i,:])
        plt.savefig(val_result_dir+"/val_data_"+str(i)+".png")
        plt.close("all")
        
if __name__=="__main__":
    main()