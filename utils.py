import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from tensorflow.python.keras.losses import LossFunctionWrapper, categorical_crossentropy
from tensorflow.python.keras.utils import losses_utils
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score


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


def load_npy_binary(path):
    """Loads in a Numpy binary.
      Args:
        path: Path to Numpy binary.
      Returns:
        A Numpy array.
      """
    return np.load(path)


def masked_categorical_cross_entropy():
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = tf.keras.losses.CategoricalCrossentropy()
        l = loss(y_true, y_pred)

        return l

    return loss


def masked_categorical_cross_entropy_test():
    # mask = K.variable()
    strategy = tf.distribute.MirroredStrategy()
    def loss(y_true, y_pred):
        with strategy.scope():
            kerasloss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            l = kerasloss(y_true, y_pred)
        return l

    return loss


def expand_dim(low_dim_tensor, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.
      Args:
        low_dim_tensor: List of tensors.
      Returns:
        A tensor.
      """
    return K.stack(low_dim_tensor, axis=axis)


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
    distance_maps = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))

    bins = np.linspace(min_angstrom, max_angstrom, num_bins)
    if len(output.shape) == 4:
        values = np.argmax(output, axis=3)
    else:
        values = np.argmax(output, axis=2)
    for batch in range(distance_maps.shape[0]):
        distance_maps[batch] = bins[values[batch]]

    return distance_maps


def pad_tensor(tensor, shape):
    if isinstance(shape, int):
        shape = tuple([shape])
    else:
        shape = tuple(shape)
    dim = len(shape)
    padded_tensor = np.zeros(shape)
    if dim == 1:
        padded_tensor[0:tensor.shape[0]] = tensor
    elif dim == 2:
        padded_tensor[0:tensor.shape[0], 0:tensor.shape[0]] = tensor

    return padded_tensor


def pad_primary(tensor, shape):
    #AA space is betwen 0-19 --> paddings have Id 20
    curr_length = tensor.shape[0]
    padded_tensor = np.full(shape, 20)
    padded_tensor[0:curr_length] = tensor

    return padded_tensor


def pad_tertiary(tensor, shape):
    curr_length = tensor.shape[0]
    padded_tensor = np.zeros(shape=shape)
    padded_tensor[0:curr_length, 0:curr_length] = tensor

    return padded_tensor


def pad_mask(tensor, shape):
    curr_length = tensor.shape[0]
    padded_tensor = np.zeros(shape=shape)
    padded_tensor[0:curr_length, 0:curr_length] = tensor

    return padded_tensor


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
    for batch in range(distance_maps.shape[0]):
        contact_maps[batch] = np.where(distance_maps[batch] > 8, 0, 1)  # Distance > 8 yield 0, otherwise 1

    return contact_maps


def distogram_accuracy_metric(y_true, y_pred, mask, minimum_bin_val, maximum_bin_val, num_bins):
     """Computes the individual accuracies and mean accuracy for a batch of predictions
     based on the predicted dostograms.
      Args:
        y_true: Batch of ground truths.
        y_pred: Batch of predictions.
        mask: Batch of masks.
      Returns:
        List with accuracies for each prediction in batch and mean accuracy for batch of predictions.
      """
     distance_maps_pred = output_to_distancemaps(y_pred, minimum_bin_val, maximum_bin_val, num_bins)
     total_accu = 0
     set_size = y_true.shape[0]
     sample_acc = np.zeros((y_true.shape[1], y_true.shape[2]))
     samples_acc = []
     for sample in range(y_true.shape[0]):
          y_pred_disto = to_distogram(distance_maps_pred[sample], minimum_bin_val, maximum_bin_val, num_bins)
          for x in range(y_true[sample].shape[0]):
              for y in range(y_true[sample].shape[1]):
                  aa_pair_accuracy = accuracy_score(y_true[sample][x,y], y_pred_disto[x,y])
                  sample_acc[x,y] = aa_pair_accuracy
          sample_accuracy = (sample_acc.flatten() * mask[sample].flatten()).sum()/np.count_nonzero(mask[sample])
          if (math.isnan(sample_accuracy)):
              set_size  = set_size - 1
              continue
          samples_acc.append(sample_accuracy)
          total_accu = total_accu + sample_accuracy

     return samples_acc, total_accu / set_size


def distogram_precision_metric(y_true, y_pred, mask, minimum_bin_val, maximum_bin_val, num_bins):
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
     set_size = y_true.shape[0]
     true_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     pred_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     samples_prec = []
     classes = [str(i) for i in range(num_bins)]   ##[0-->63]
     for sample in range(y_true.shape[0]):
          y_pred_disto = to_distogram(distance_maps_pred[sample], minimum_bin_val, maximum_bin_val, num_bins)
          for x in range(y_true[sample].shape[0]):
              for y in range(y_true[sample].shape[1]):
                  bin_index_true = np.argmax(y_true[sample][x,y])
                  bin_index_pred = np.argmax(y_pred_disto[x,y])
                  y_true_class = classes[bin_index_true]
                  y_pred_class = classes[bin_index_pred]
                  true_classes[x,y] = y_true_class
                  pred_classes[x,y] = y_pred_class
          sample_precision = precision_score(true_classes.flatten(), pred_classes.flatten(),
                                            average = 'micro', sample_weight= mask[sample].flatten())
          if (math.isnan(sample_precision)):
              set_size  = set_size - 1
              continue
          samples_prec.append(sample_precision)
          total_precesion = total_precesion + sample_precision

     return samples_prec, total_precesion / set_size


def distogram_recall_metric(y_true, y_pred, mask, minimum_bin_val, maximum_bin_val, num_bins):
     """Computes the individual recalls and mean recall for a batch of predictions
     based on the predicted distograms.
     Args:
     y_true: Batch of ground truths.
     y_pred: Batch of predictions.
     mask: Batch of masks.
     Returns:
     List with recalls for each prediction in batch and mean recall for batch of predictions.
     """
     distance_maps_pred = output_to_distancemaps(y_pred, minimum_bin_val, maximum_bin_val, num_bins)
     total_recall = 0
     set_size = y_true.shape[0]
     true_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     pred_classes = np.zeros((y_true.shape[1], y_true.shape[2]))
     samples_recall = []
     classes = [str(i) for i in range(num_bins)]   # [0 --> 63]
     for sample in range(y_true.shape[0]):
          y_pred_disto = to_distogram(distance_maps_pred[sample], minimum_bin_val, maximum_bin_val, num_bins)
          for x in range(y_true[sample].shape[0]):
              for y in range(y_true[sample].shape[1]):
                  bin_index_true = np.argmax(y_true[sample][x,y])
                  bin_index_pred = np.argmax(y_pred_disto[x,y])
                  y_true_class = classes[bin_index_true]
                  y_pred_class = classes[bin_index_pred]
                  true_classes[x,y] = y_true_class
                  pred_classes[x,y] = y_pred_class
          sample_recall = recall_score(true_classes.flatten(), pred_classes.flatten(),
                                             average = 'micro', sample_weight= mask[sample].flatten())
          if (math.isnan(sample_recall)):
              set_size  = set_size - 1
              continue
          samples_recall.append(sample_recall)
          total_recall = total_recall + sample_recall

     return samples_recall, total_recall / set_size


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
     distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distancemap(distance_maps_predicted)
     contact_maps_true = contact_map_from_distancemap(distance_maps_true)
     set_size =  contact_maps_true.shape[0]
     total_accu = 0
     sample_acc = []
     for sample in range(contact_maps_true.shape[0]):
         sample_accuracy = accuracy_score(contact_maps_true[sample].flatten(), contact_maps_predicted[sample].flatten(),
                                          normalize=False,
                                          sample_weight=mask[sample].flatten()) / np.count_nonzero(mask[sample])
         if (math.isnan(sample_accuracy)):
             set_size  = set_size - 1
             continue
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
     distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distancemap(distance_maps_predicted)
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
             set_size =  set_size - 1
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
    distance_maps_predicted = output_to_distancemaps(y_pred, 2, 22, 64)
    distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
    contact_maps_predicted = contact_map_from_distancemap(distance_maps_predicted)
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
        sample_rec = true_pos / (true_pos + false_neg)
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

    return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)


def pad_feature2(feature, crop_size, padding_value, padding_size, rank_threshold):
    padding = tf.constant([[0, padding_size]])
    empty = tf.constant([[0, 0]])
    rank = tf.rank(feature).numpy()
    use_rank = 0
    if rank > rank_threshold:
        use_rank = rank_threshold
    else:
        use_rank = rank
    # print(use_rank)
    padding = tf.repeat(padding, use_rank, axis=0)
    for _ in range(rank-use_rank):
        padding = tf.concat([padding, empty], 0)
    feature = tf.pad(feature, padding, constant_values=tf.cast(padding_value, feature.dtype))
    return feature
