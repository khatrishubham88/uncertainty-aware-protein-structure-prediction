import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import *
from readData_from_TFRec import parse_test_dataset, parse_dataset


if __name__=="__main__":
    testdata_path1 = glob.glob("/home/ghalia/Documents/alphafold/casp7/training/100/*")
    testdata_path2 = glob.glob("/home/ghalia/Documents/alphafold/casp8/testing/*")
    testdata_path3 = glob.glob("/home/ghalia/Documents/alphafold/casp9/testing/*")
    testdata_path4 = glob.glob("/home/ghalia/Documents/alphafold/casp10/testing/*")
    dis1 = []
    dis2 = []
    dis3 = []
    dis4 = []
    proteins = 0
    ax = plt.subplot()
    for primary, evolutionary, tertiary, ter_mask in parse_dataset(testdata_path1):
        if (primary != None):
            dist_map = calc_pairwise_distances(tertiary)
            dist_map = np.asarray(dist_map)
            dis1.extend(dist_map.flatten())
            proteins = proteins +1
    # for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path2, 3):
    #     if (primary != None):
    #         dist_map = calc_pairwise_distances(tertiary)
    #         dist_map = np.asarray(dist_map)
    #         dis2.extend(dist_map.flatten())
    #         proteins = proteins +1
    # for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path3, 3):
    #     if (primary != None):
    #         dist_map = calc_pairwise_distances(tertiary)
    #         dist_map = np.asarray(dist_map)
    #         dis3.extend(dist_map.flatten())
    #         proteins = proteins +1
    # for primary, evolutionary, tertiary, ter_mask in parse_test_dataset(testdata_path4, 3):
    #     if (primary != None):
    #         dist_map = calc_pairwise_distances(tertiary)
    #         dist_map = np.asarray(dist_map)
    #         dis4.extend(dist_map.flatten())
    #         proteins = proteins +1

    z1 = np.count_nonzero(dis1)
    #z2 = np.count_nonzero(dis2)
    #z3 = np.count_nonzero(dis3)
    #z4 = np.count_nonzero(dis4)
    #non_zeros = z1+z2+z3+z4
    #ax.hist([dis1,dis2,dis3,dis4], bins = 25, range=(0,150), rwidth=0.7, label=['casp7','casp8','casp9','casp10'], stacked=False)
    #ax.hist([dis1,dis2,dis3], bins = 25, range=(0,150), rwidth=0.7, label=['TBM','FM','TBM-Hard'], stacked=False)
    ax.hist(dis1, bins = 20, range=(0,150), rwidth=0.7, label=['casp7'], stacked=False)
    y_vals = ax.get_yticks()
    ax.set_yticklabels(["{:.1f}%".format(math.ceil((x / z1)*100)) for x in y_vals])

    plt.title('Casp-7 Distance Histogrm, #proteins='+str(proteins))
    plt.xlabel('Distance [Angs]')
    plt.ylabel('Proportion [%]')
    plt.legend()
    plt.savefig("casp7-all.png")
