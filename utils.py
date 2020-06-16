import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
# from readData_from_TFRec import widen_seq
from tensorflow.python.keras.losses import LossFunctionWrapper, categorical_crossentropy
from tensorflow.python.keras.utils import losses_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
tf.compat.v1.enable_eager_execution()


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


def expand_dim(low_dim_tensor):
    return K.stack(low_dim_tensor, axis=0)


def output_to_distancemaps(output, min_angstrom, max_angstrom, num_bins):
    output = K.eval(output)
    distance_maps = np.zeros(shape=(output.shape[0], output.shape[1], output.shape[2]))

    bins = np.linspace(min_angstrom, max_angstrom, num_bins)
    values = np.argmax(output, axis=3)
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
    """Calculates the distance between two AA
    using three different approaches:
        1- distance between the N atoms of the two AA
        2- distance between the C-alpha atoms of the two AA
        3- distance between the C-prime atoms of the two AA
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
    """This function takes as input the two C-alpha corrdinates
    of two AAs and returns the angstrom distance between them
    input: coord1 [x, y, z], coord2 [x, y, z]
    """
    C_alpha_distance = math.sqrt(
        (coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2 + (coord2[2] - coord1[2]) ** 2)

    return C_alpha_distance


def calc_pairwise_distances(tertiary):
    """Calculates the pairwise distances between
    AAs of a protein and returns the distance map in angstrom.
    input: tertiary is a tensor of shape (seq_len, 3)
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
    """Returns the distogram tensor LxLxnum_bins from the distance map LxL.
    Input: distance_map: LxL distance matrx in angstrom
           min: minimum value for the histogram in angstrom
           max: maximum value for the histogram in angstrom
           num_bins: integer number
    """
    assert min_val >= 0.0
    assert max_val > 0.0
    histo_range = max_val-min_val

    distance_map = np.clip(distance_map, a_min=min_val, a_max=max_val)
    distance_map = np.int32(np.floor((num_bins - 1)*(distance_map-min_val)/(histo_range)))
    distogram = np.eye(num_bins)[distance_map]

    return distogram


def random_index(primary, crop_size):
    """
    This function returns a random index to do the cropping,
    index is within sequence lengh if seq_len > crop_size
    """
    index = []
    if primary.shape[0] <= crop_size:
        index.extend([0, 0])
    else:
        index.extend([np.random.randint(0, primary.shape[0] - crop_size),
                      np.random.randint(0, primary.shape[0] - crop_size)])
    return index


def contact_map_from_distancemap(distance_maps):
  """if distance between 2 AA is smaller than 8 Angstrom set to contact
  otherwise not in contact:
  input: distance_maps of shape [nr_samples, 64, 64]
  output: contact_maps of shape [nr_samples, 64, 64]
  """
  contact_maps = np.zeros(shape=(distance_maps.shape[0], distance_maps.shape[1], distance_maps.shape[2]))
  for batch in range(distance_maps.shape[0]):
      contact_maps[batch] = np.where(distance_maps[batch] > 8, 0, 1) # dist >8 yield 0 otherwise 1
  return contact_maps


def accuracy_metric(y_true, y_predict):
     """
     input:
        y_predict: predicted distograms of shape [nr_samples, 64, 64, 64]
        y_true: ground truth distograms of shape [nr_samples, 64, 64, 64]
    output:accuracy using contact maps
     """
     distance_maps_predicted = output_to_distancemaps(y_predict, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distancemap(distance_maps_predicted)
     contact_maps_true = contact_map_from_distancemap(distance_maps_true)
     total_accu = 0
     for sample in range(contact_maps_true.shape[0]):
         sample_accu = accuracy_score(contact_maps_true[sample].flatten(), contact_maps_predicted[sample].flatten())
         total_accu = total_accu + sample_accu
     return (total_accu/contact_maps_true.shape[0])


def precision_metric(y_true, y_predict):
     """
     input:
        y_predict: predicted distograms of shape [nr_samples, 64, 64, 64]
        y_true: ground truth distograms of shape [nr_samples, 64, 64, 64]
    output:presicion using contact maps
     """
     #print(type(y_true))
     print(type(y_predict))
     #print(y_true.shape)
     #print(y_predict.sha)
     distance_maps_predicted = output_to_distancemaps(y_predict, 2, 22, 64)
     distance_maps_true = output_to_distancemaps(y_true, 2, 22, 64)
     contact_maps_predicted = contact_map_from_distancemap(distance_maps_predicted)
     contact_maps_true = contact_map_from_distancemap(distance_maps_true)
     total_prec = 0
     for sample in range(contact_maps_true.shape[0]):
         sample_prec = precision_score(contact_maps_true[sample].flatten(), contact_maps_predicted[sample].flatten())
         total_prec = total_prec + sample_prec
     return (total_prec/contact_maps_true.shape[0])


def pad_feature2(feature, crop_size, padding_value, padding_size, rank_threshold):
    padding = tf.constant([[0, padding_size]])
    empty = tf.constant([[0, 0]])
    rank = tf.rank(feature).numpy()
    use_rank = 0
    if rank>rank_threshold:
        use_rank = rank_threshold
    else:
        use_rank = rank
    # print(use_rank)
    padding = tf.repeat(padding, use_rank, axis=0)
    for _ in range(rank-use_rank):
        padding = tf.concat([padding, empty], 0)
    feature = tf.pad(feature, padding, constant_values=tf.cast(padding_value, feature.dtype))
    return feature
