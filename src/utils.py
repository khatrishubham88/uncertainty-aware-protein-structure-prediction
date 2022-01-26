import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import warnings

from tensorflow.python.keras.losses import LossFunctionWrapper, categorical_crossentropy
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.metrics import CategoricalAccuracy
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, f1_score


class CategoricalCrossentropyForDistributed(LossFunctionWrapper):
    def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_crossentropy',
               global_batch_size=None):
        if global_batch_size is None:
            raise ValueError("Global batch size must be provided")
        if not isinstance(global_batch_size, int):
            raise ValueError("Global batch size must be a integer")
        global_batch_size = tf.constant(global_batch_size, dtype=tf.float32)
        super(CategoricalCrossentropyForDistributed, self).__init__(
        categorical_crossentropy_with_wrapper,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        global_batch_size=global_batch_size)


def categorical_crossentropy_with_wrapper(y_true, y_pred, global_batch_size, from_logits=False, label_smoothing=0):
    loss = categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=from_logits, label_smoothing=label_smoothing)
    # weight by global batch size
    return loss/global_batch_size


def get_batch_metric(metric, true, predict, mask):
    batch_acc = []
    for elem in range(int(true.shape[0])):
        metric.reset_states()
        _ = metric.update_state(true[elem], predict[elem], sample_weight=mask[elem])
        batch_acc.append(metric.result().numpy())

    return batch_acc


def mc_accuracy(y_true, mc_mean_y_pred, mask):
    accs = []
    m = CategoricalAccuracy()
    if len(mc_mean_y_pred.shape)==len(y_true.shape)+1:
        for i in range(mc_mean_y_pred.shape[0]):
            accs.append(get_batch_metric(m, y_true, mc_mean_y_pred[i], mask))
    elif len(mc_mean_y_pred.shape)==len(y_true.shape):
        accs = get_batch_metric(m, y_true, mc_mean_y_pred, mask)
    else:
        raise ValueError("Inappropriate shape of predicted sample")
    del m

    return accs


def mc_hist_plot(fname, metric_data, mean_acc=None, title="Accuracy distribution"):
    plt.figure()
    plt.title(title)
    plt.hist(metric_data)
    if mean_acc is not None:
        plt.axvline(x=mean_acc, color="b")
    plt.savefig(fname)
    # plt.close("all")


def distance_map_plotter(fname, y_true, y_pred, mask, title="Distancemap Plots"):
    plt.figure()
    plt.subplot(131)
    plt.title("Ground Truth")
    plt.imshow(y_true, cmap='viridis_r')
    plt.subplot(132)
    plt.title("Prediction by model")
    plt.imshow(y_pred, cmap='viridis_r')
    plt.subplot(133)
    plt.title("mask")
    plt.imshow(mask, cmap='viridis_r')
    plt.suptitle(title, fontsize=16)
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        left=False,
        right=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False)
    plt.axis('off')
    plt.savefig(fname)
    # plt.close("all")


def mc_distance_map_plotter(fname, y_true, y_pred_mean, y_pred_best, mask, title="Distancemap Plots"):
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("Ground Truth")
    plt.imshow(y_true, cmap='viridis_r')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        left=False,
        right=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False) # labels along the bottom edge are off
    plt.axis('off')
    plt.subplot(222)
    plt.title("Mean Prediction")
    plt.imshow(y_pred_mean, cmap='viridis_r')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        left=False,
        right=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False) # labels along the bottom edge are off
    plt.axis('off')
    plt.subplot(223)
    plt.title("Best Prediction")
    plt.imshow(y_pred_best, cmap='viridis_r')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        left=False,
        right=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False) # labels along the bottom edge are off
    plt.axis('off')
    plt.subplot(224)
    plt.title("mask")
    plt.imshow(mask, cmap='viridis_r')
    plt.suptitle(title, fontsize=16)
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        direction='inout',
        left=False,
        right=False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False) # labels along the bottom edge are off
    plt.axis('off')
    plt.savefig(fname)

def get_class_weights(dataset, num_steps, channels, path=None):
  cw={}
  count = 0
  batch_sum = tf.zeros(channels)
  # find total occurance for each class
  for i, (_, y,_) in enumerate(dataset):
    batch_sum += tf.math.reduce_sum(y, axis=list(range(len(y.shape)-1)))
    count += np.prod(y.shape[:-1])
    if i == num_steps:
      break
  
  # check non zero minimum occurance to deal with case of 0 occurences of some class
  nnz_mask = tf.math.greater(batch_sum, 0).numpy()
  zero_mask = tf.math.logical_not(nnz_mask)
  nnz_min_val =  tf.math.reduce_min(batch_sum[nnz_mask]).numpy()
  batch_sum = tf.where(zero_mask, x=nnz_min_val, y=batch_sum) # replacing 0 with minimum non 0 occurence
  # increment the count to account for correction and have sum of probability = 1 condition
  count += int((tf.math.reduce_sum(tf.cast(zero_mask, dtype=tf.int32)).numpy())*nnz_min_val)
  
  # biased probablities
  batch_sum /= count
  # sanctity check
  if float(tf.math.reduce_sum(batch_sum).numpy()) - 1.0 > 1e-12:
    raise ValueError("Sum of probability is not 1.0 check the input arrays!")
  
  weights = (tf.math.reciprocal(batch_sum)*(1.0/float(batch_sum.shape[0]))).numpy()
  # normalization factor
  min_val = float(tf.math.reduce_min(weights).numpy())
  
  for i in range(channels):
    cw[i] = float(weights[i])/min_val
  
  if path is not None:
      with open(path, 'wb') as handle:
          pickle.dump(cw, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  return cw

def load_npy_binary(path):
    """Loads in a Numpy binary.
      Args:
        path: Path to Numpy binary.
      Returns:
        A Numpy array.
      """
    return np.load(path)


def expand_dim(low_dim_tensor, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.
      Args:
        low_dim_tensor: List of tensors.
        axis:           Index of axis where dimension should be added.
      Returns:
        A tensor.
      """
    return K.stack(low_dim_tensor, axis=axis)


def create_protein_batches(padded_primary, padded_evol, padded_dist_map, padded_mask, crop_size, stride, min_bin_val,
                           max_bin_val, num_bins):
    """Given the padded primary and evolutionary features as well as the padded ground
    truth and mask, this function creates batches of testing data for the test pipeline.
      Args:
         padded_primary:  Numpy array containing the padded primary information for the
                          input to the network.
         padded_evol:     Numpy array containing the padded evolutionary information for
                          the input to the network.
         padded_dist_map: Numpy array containing the padded ground truth as distance map.
         padded_mask:     Numpy array containing the padded masking tensor.
         crop_size:       Number of amino acids in a crop as integer.
         stride:          Stride value for cropping as integer.
         min_bin_val:     Lower bound of distance range [Angstrom] for distance prediction
                          as integer (here: 2).
         max_bin_val:     Upper bound of distance range [Angstrom] for distance prediction
                          as integer (here: 22).
         num_bins:        Number of bins used for discretization of distance range as
                          integer (here: 64).
      Returns:
         List of tuples containing input crop, ground truth crop and masking tensor crop.
    """
    batches = []
    for x in range(0, padded_primary.shape[0] - crop_size, stride):
        for y in range(0, padded_primary.shape[0] - crop_size, stride):
            primary_2D_crop = padded_primary[x:x + crop_size, y:y + crop_size, :]
            pssm_crop = padded_evol[x:x + crop_size, y:y + crop_size, :]
            pri_evol_crop = tf.concat([primary_2D_crop, pssm_crop], axis=2)
            tertiary_crop = padded_dist_map[x:x + crop_size, y:y + crop_size]
            tertiary_crop = to_distogram(tertiary_crop, min_bin_val, max_bin_val, num_bins)
            mask_crop = padded_mask[x:x + crop_size, y:y + crop_size]
            batches.append((pri_evol_crop, tertiary_crop, mask_crop))

    return batches


def output_to_distancemaps(output, min_angstrom, max_angstrom, num_bins):
    """Given a batch of outputs, creates the distance maps ready for plotting.
      Args:
        output: Batch of predictions by the network either as tensor or Numpy array.
        min_angstrom: Lower boundary of distance range in Angstrom.
        max_angstrom: Upper boundary of distance range in Angstrom.
        num_bins: Number of bins for discretization of the distance range.
      Returns:
        A Numpy array with consisting of predicted distances for each residual pair in the crop.
    """
    output = K.eval(output)

    bins = np.linspace(min_angstrom, max_angstrom, num_bins)
    if len(output.shape) == 4:
        values = np.argmax(output, axis=3)
        # print(values)
        distance_maps = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))
        for batch in range(distance_maps.shape[0]):
          distance_maps[batch] = bins[values[batch]]
    else:
        values = np.argmax(output, axis=2)
        distance_maps = np.zeros(shape=(output.shape[0], output.shape[1]))
        distance_maps = bins[values]

    return distance_maps


def output_to_distogram(output, min_angstrom, max_angstrom, num_bins):
    """Given a batch of outputs, creates the distance maps ready for plotting.
      Args:
        output: Batch of predictions by the network either as tensor or Numpy array.
        min_angstrom: Lower boundary of distance range in Angstrom.
        max_angstrom: Upper boundary of distance range in Angstrom.
        num_bins: Number of bins for discretization of the distance range.
      Returns:
        A Numpy array with consisting of predicted distances for each residual pair in the crop.
    """
    bins = np.linspace(min_angstrom, max_angstrom, num_bins)
    if len(output.shape) == 4:
        values = np.argmax(output, axis=3) # 3D
        distogram = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2], output.shape[3]))
        for batch in range(distogram.shape[0]):
          distogram[batch] = bins[values[batch]]
    else:
        values = np.argmax(output, axis=2)
        distogram = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))
        distance_maps = bins[values]

    return distance_maps


def calc_distance(aa1, aa2):
    """Calculates the distance between two amino acids using
    three different approaches:
    1. Distance between the N atoms of the two amino acids.
    2. Distance between the C-Alpha atoms of the two amino acids.
    3. Distance between the C-Prime atoms of the two amino acids.
      Args:
        aa1: 3D Coordinates of atoms in first amino acid.
        aa2: 3D Coordinates of atoms in second amino acid.
      Returns:
        Lists of pairwise distances between amino acids using N, C-Alpha
        or C-Prime atoms for calculation.
    """
    aa1_N_pos = (aa1[0][0], aa1[0][1], aa1[0][2])
    aa2_N_pos = (aa2[0][0], aa2[0][1], aa2[0][2])

    aa1_C_alpha_pos = (aa1[1][0], aa1[1][1], aa1[1][2])
    aa2_C_alpa_pos = (aa2[1][0], aa2[1][1], aa2[1][2])

    aa1_C_prime_pos = (aa1[2][0], aa1[2][1], aa1[2][2])
    aa2_C_prime_pos = (aa2[2][0], aa2[2][1], aa2[2][2])

    N_distance = math.sqrt(
        (aa2_N_pos[0] - aa1_N_pos[0]) ** 2 + (aa2_N_pos[1] - aa1_N_pos[1]) ** 2 + (aa2_N_pos[2] - aa1_N_pos[2]) ** 2)
    C_alpha_distance = math.sqrt(
        (aa2_C_alpa_pos[0] - aa1_C_alpha_pos[0]) ** 2 + (aa2_C_alpa_pos[1] - aa1_C_alpha_pos[1]) ** 2 + (
                    aa2_C_alpa_pos[2] - aa1_C_alpha_pos[2]) ** 2)
    C_prime_distance = math.sqrt(
        (aa2_C_prime_pos[0] - aa1_C_prime_pos[0]) ** 2 + (aa2_C_prime_pos[1] - aa1_C_prime_pos[1]) ** 2 + (
                    aa2_C_prime_pos[2] - aa1_C_prime_pos[2]) ** 2)

    return N_distance, C_alpha_distance, C_prime_distance


def calc_calpha_distance(coord1, coord2):
    """This function calculates the distance between two amino acids in Angstrom
    using the coordinates of the C-Alpha atoms for calculation.
      Args:
        coord1: 3D coordinates of first C-Alpha atom [x, y, z].
        coord2: 3D coordinates of second C-Alpha atom [x, y, z].
      Returns:
        Distance between both C-Alpha atoms in Angstrom.
    """
    C_alpha_distance = math.sqrt(
        (coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2 + (coord2[2] - coord1[2]) ** 2)

    return C_alpha_distance


def calc_pairwise_distances(tertiary):
    """Calculates the pairwise distances between amino acids of a
    protein and returns the distance map in Angstrom.
      Args:
        tertiary: Tensor containing position information of each amino
                  acid in protein.
      Returns:
        Tensor containing pairwise distances between amino acids in protein.
    """
    tertiary_numpy = tertiary.numpy() / 100
    c_alpha_coord = []
    for index, coord in enumerate(tertiary_numpy):
        # Extract only c-alpha coordinates
        if (index % 3 == 1):
            c_alpha_coord.append(coord.tolist())

    # Initialize the distance matrix of shape (len_seq, len_seq)
    distance_matrix = []
    for i, coord1 in enumerate(c_alpha_coord):
        dist = []
        for j, coord2 in enumerate(c_alpha_coord):
            distance = calc_calpha_distance(coord1, coord2)
            dist.append(distance)
        distance_matrix.append(dist)

    return tf.convert_to_tensor(distance_matrix)


def to_distogram(distance_map, min_val, max_val, num_bins):
    """This function returns the distogram tensor LxLxNum_bins from
    the distance map LxL.
      Args:
        distance_map: LxL distance matrix in Angstrom.
        min: Minimum value for the histogram in Angstrom
        max: Maximum value for the histogram in Angstrom
        num_bins: Number of bins used for discretization of distance range.
      Returns:
        Numpy array containing corresponding distogram for given distance map.
    """
    assert min_val >= 0.0
    assert max_val > 0.0
    histo_range = max_val-min_val

    distance_map = np.clip(distance_map, a_min=min_val, a_max=max_val)
    distance_map = np.int32(np.floor((num_bins - 1)*(distance_map-min_val)/(histo_range)))
    distogram = np.eye(num_bins)[distance_map]

    return distogram


def random_index(primary, crop_size):
    """This function returns a random index to do the cropping,
    index is within sequence length if seq_len is bigger than crop_size.
      Args:
        primary: Tensor containing information about protein sequence.
        crop_size: Number of residual pairs in one crop.
      Returns:
        List with two indices (x and y dimensions) for random cropping.
    """
    index = []
    if primary.shape[0] <= crop_size:
        index.extend([0, 0])
    else:
        index.extend([np.random.randint(0, primary.shape[0] - crop_size),
                      np.random.randint(0, primary.shape[0] - crop_size)])
    return index


def contact_map_from_distancemap(distance_maps):
    """If distance between two amino acids is smaller than 8 Angstrom, set to contact
    otherwise not in contact:
      Args:
        distance_maps: Numpy array containing batch of distance maps.
      Returns:
        Numpy array containing batch of corresponding contact maps.
    """
    contact_maps = np.zeros(shape=(distance_maps.shape[0], distance_maps.shape[1], distance_maps.shape[2]))
    for sample in range(distance_maps.shape[0]):
        contact_maps[sample] = np.where(distance_maps[sample] > 8, 0, 1)  # Distance > 8 yield 0, otherwise 1

    return contact_maps


def contact_map_from_distogram(y_predict):
    """If distance between two amino acids is smaller than 8 Angstrom, set to contact
    otherwise not in contact:
      Args:
        y_predict: output of model.predict() of shape [#samples, 64, 64, 64].
      Returns:
        Numpy array containing batch of corresponding contact maps.
    """
    contact_maps = np.zeros(shape=(y_predict.shape[0], y_predict.shape[1], y_predict.shape[2]))
    distance_maps = output_to_distancemaps(y_predict, 2, 22, 64)
    for sample in range(y_predict.shape[0]):
        for x in range(y_predict.shape[1]):
            for y in range(y_predict.shape[2]):
                density = np.sum(y_predict[sample][x,y][0:19])
                contact_maps[sample][x,y] = 1 if density > 0.5 else 0

    return contact_maps


def prob_to_class(y_pred, num_classes):
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    return tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes)


def entropy_func(y_predict):
    """Calculates entropy on a data of shape (#samples, 64, 64, 64)
        Args:
            y_predict output of model.predict()
        Returns:
            Entropy meaned across one sample and across all samples
    """
    sample_entropy = np.zeros((y_predict.shape[1], y_predict.shape[2]))
    samples_entropy = []
    tot_entropy = 0
    for sample in range(y_predict.shape[0]):
        for x in range(y_predict[sample].shape[0]):
            for y in range(y_predict[sample].shape[1]):
                ent = entropy(y_predict[sample][x,y])
                if np.isfinite(ent):
                    sample_entropy[x,y] = ent
                else:
                    warnings.warn("Infinite Entropy")
                    sample_entropy[x,y] = 0

        sample_mean = np.mean(sample_entropy)
        samples_entropy.append(sample_mean)

    return np.mean(sample_entropy)


def distogram_metrics(y_true, y_pred, mask, minimum_bin_val, maximum_bin_val, num_bins, single_sample = False):
     """Computes the individual precisions and mean precision for a batch of predictions
     based on the predicted dostograms.
      Args:
        y_true: Batch of ground truths.
        y_pred: Batch of predictions.
        mask: Batch of masks.
      Returns:
        List with precisions for each prediction in batch and mean precision for batch of predictions.
      """
     distance_maps_pred = output_to_distancemaps(y_pred, minimum_bin_val, maximum_bin_val, num_bins)
     total_precesion = 0
     total_recall = 0
     total_accuracy = 0
     total_f1 = 0
     set_size = y_true.shape[0]
     true_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     pred_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     samples_prec = []
     samples_recall = []
     samples_accuracy = []
     samples_f1 = []
     t_c = []
     p_c = []
     classes = [str(i) for i in range(num_bins)]   ##[0-->63]
     if not single_sample:
        for sample in range(y_true.shape[0]):
            y_pred_disto = to_distogram(distance_maps_pred[sample], minimum_bin_val, maximum_bin_val, num_bins)
            for x in range(y_true[sample].shape[0]):
                for y in range(y_true[sample].shape[1]):
                    bin_index_true = np.argmax(y_true[sample][x, y])
                    bin_index_pred = np.argmax(y_pred_disto[x, y])
                    y_true_class = classes[bin_index_true]
                    y_pred_class = classes[bin_index_pred]
                    true_classes[x,y] = y_true_class
                    pred_classes[x,y] = y_pred_class
            if (np.count_nonzero(mask[sample]) == 0 ):
                set_size  = set_size - 1
                continue
            sample_precision = precision_score(true_classes.flatten(), pred_classes.flatten(),
                                            average = 'weighted', sample_weight= mask[sample].flatten())
            sample_recall = recall_score(true_classes.flatten(), pred_classes.flatten(),
                                                average = 'weighted', sample_weight= mask[sample].flatten())
            sample_f1 = f1_score(true_classes.flatten(), pred_classes.flatten(),
                                                average = 'weighted', sample_weight= mask[sample].flatten())
            sample_accuracy = accuracy_score(true_classes.flatten(), pred_classes.flatten(), normalize = True,
                                                    sample_weight= mask[sample].flatten())
            t_c.extend(true_classes.flatten())
            p_c.extend(pred_classes.flatten())

            samples_prec.append(sample_precision)
            samples_recall.append(sample_recall)
            samples_accuracy.append(sample_accuracy)
            samples_f1.append(sample_f1)
            total_recall = total_recall + sample_recall
            total_precesion = total_precesion + sample_precision
            total_accuracy = total_accuracy + sample_accuracy
            total_f1 = total_f1 + sample_f1
     else:
         y_pred_disto = to_distogram(distance_maps_pred, minimum_bin_val, maximum_bin_val, num_bins)
         for x in range(y_true.shape[0]):
             for y in range(y_true.shape[1]):
                 bin_index_true = np.argmax(y_true[x, y])
                 bin_index_pred = np.argmax(y_pred_disto[x, y])
                 y_true_class = classes[bin_index_true]
                 y_pred_class = classes[bin_index_pred]
                 true_classes[x,y] = y_true_class
                 pred_classes[x,y] = y_pred_class
         if (np.count_nonzero(mask) == 0 ):
             return 0,0,0,0, None

         total_recall = recall_score(true_classes.flatten(), pred_classes.flatten(),
                                                average = 'weighted', sample_weight= mask.flatten())
         total_precesion = precision_score(true_classes.flatten(), pred_classes.flatten(),
                                            average = 'weighted', sample_weight= mask.flatten())
         sample_recall = recall_score(true_classes.flatten(), pred_classes.flatten(),
                                            average = 'weighted', sample_weight= mask.flatten())
         total_f1 = f1_score(true_classes.flatten(), pred_classes.flatten(),
                                            average = 'weighted', sample_weight= mask.flatten())
         total_accuracy = accuracy_score(true_classes.flatten(), pred_classes.flatten(), normalize = True,
                                                sample_weight= mask.flatten())
         set_size = 1
     cm = confusion_matrix(t_c, p_c)

     return total_accuracy/set_size, total_precesion/set_size, total_recall/set_size, total_f1/set_size, cm


def accuracy_metric(y_true, y_pred, mask):
     """Computes the individual accuracies and mean accuracy for a batch of predictions
     based on the predicted contact maps.
      Args:
        y_true: Batch of ground truths.
        y_pred: Batch of predictions.
        mask: Batch of masks.
      Returns:
        List with accuracies for each prediction in batch and mean accuracy for batch of predictions.
      """
     #distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distogram(y_pred)
     contact_maps_true = contact_map_from_distancemap(distance_maps_true)
     set_size =  contact_maps_true.shape[0]
     total_accu = 0
     sample_acc = []
     for sample in range(contact_maps_true.shape[0]):
         if (np.count_nonzero(mask[sample]) == 0 ):
             set_size  = set_size - 1
             continue
         sample_accuracy = accuracy_score(contact_maps_true[sample].flatten(), contact_maps_predicted[sample].flatten(),
                                          normalize=True,
                                          sample_weight=mask[sample].flatten())
         sample_acc.append(sample_accuracy)
         total_accu = total_accu + sample_accuracy
     return sample_acc, total_accu / set_size


def precision_metric(y_true, y_pred, mask):
     """Computes the individual predicions and mean precision for a batch of predictions
     based on the predicted contact maps.
      Args:
        y_true: Batch of ground truths.
        y_pred: Batch of predictions.
        mask: Batch of masks.
      Returns:
        List with precisions for each prediction in batch and mean precision for batch of predictions.
      """
     #distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distogram(y_pred)
     contact_maps_true = contact_map_from_distancemap(distance_maps_true)
     total_prec = 0
     precisions = []
     set_size =  contact_maps_true.shape[0]
     for sample in range(contact_maps_true.shape[0]):
         true_pos = (((contact_maps_true[sample].flatten() == contact_maps_predicted[sample].flatten())
                      & (contact_maps_true[sample].flatten() == 1)
                      & (contact_maps_predicted[sample].flatten() == 1)) * mask[sample].flatten()).sum()
         false_pos = (((contact_maps_true[sample].flatten() != contact_maps_predicted[sample].flatten())
                      & (contact_maps_true[sample].flatten() == 0)
                      & (contact_maps_predicted[sample].flatten() == 1)) * mask[sample].flatten()).sum()
         sample_prec = true_pos / (true_pos + false_pos)
         if (math.isnan(sample_prec)):
             set_size = set_size - 1
             continue
         precisions.append(sample_prec)
         total_prec = total_prec + sample_prec

     return precisions, total_prec / set_size


def recall_metric(y_true, y_pred, mask):
    """Computes the individual recall and mean recall for a batch of predictions
    based on the predicted contact maps.
      Args:
        y_true: Batch of ground truths.
        y_pred: Batch of predictions.
        mask: Batch of masks.
      Returns:
        List with recalls for each prediction in batch and mean recall for batch of predictions.
    """
    #distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
    distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
    contact_maps_predicted = contact_map_from_distogram(y_pred)
    contact_maps_true = contact_map_from_distancemap(distance_maps_true)
    total_rec = 0
    recalls = []
    set_size = contact_maps_true.shape[0]
    for sample in range(contact_maps_true.shape[0]):
        true_pos = (((contact_maps_true[sample].flatten() == contact_maps_predicted[sample].flatten())
                     & (contact_maps_true[sample].flatten() == 1)
                     & (contact_maps_predicted[sample].flatten() == 1)) * mask[sample].flatten()).sum()
        false_neg = (((contact_maps_true[sample].flatten() != contact_maps_predicted[sample].flatten())
                      & (contact_maps_true[sample].flatten() == 1)
                      & (contact_maps_predicted[sample].flatten() == 0)) * mask[sample].flatten()).sum()

        if true_pos + false_neg > 0:
            sample_rec = true_pos / (true_pos + false_neg)
        else:
            sample_rec = 0.0
        if (math.isnan(sample_rec)):
            set_size = set_size - 1
            continue
        recalls.append(sample_rec)
        total_rec = total_rec + sample_rec

    return recalls, total_rec / set_size


def f_beta_score(precision, recall, beta=1):
    """Computes the FBeta score.
      Args:
        precision: Integer or float.
        recall: Integer or float.
        beta: Beta Parameter (Beta=0.5: F0.5 Score, Beta=1: F1 Score, Beta=2: F2 Score).
      Returns:
        FBeta score as integer or float.
    """
    if (beta**2 * precision + recall) !=0:
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    else:
        return 0


def pad_feature2(feature, crop_size, padding_value, padding_size, rank_threshold):
    padding = tf.constant([[0, padding_size]])
    empty = tf.constant([[0, 0]])
    rank = tf.rank(feature).numpy()
    use_rank = 0
    if rank > rank_threshold:
        use_rank = rank_threshold
    else:
        use_rank = rank
    padding = tf.repeat(padding, use_rank, axis=0)
    for _ in range(rank-use_rank):
        padding = tf.concat([padding, empty], 0)
    feature = tf.pad(feature, padding, constant_values=tf.cast(padding_value, feature.dtype))

    return feature
