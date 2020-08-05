from tensorflow.keras import Input
from tensorflow import keras
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Softmax
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.python.ops import array_ops
import tensorflow as tf
import time
import tqdm
from utils import output_to_distancemaps
from mc_utils import *
import shutil
import math


class ResNetV2(keras.Model):
    """Two-dimensional dilated convolutional neural network with variable number of residual
    block groups. Each residual block group consists of four ResNet blocks.
    """
    @classmethod
    def none_reg(cls, val):
        return None
    # class variables
    input_channels = None
    output_channels = None
    crop_size = None
    num_blocks = None
    num_channels = None
    dilation = None
    batch_size = None
    non_linearity = None
    dropout_rate = None
    reg_strength = None
    kernel_initializer = None
    logits = None
    kernel_regularizer = None
    sparse = None
    first_layers = None
    resnet_stack = None
    dropout_layers = None
    identity_add_layer = None
    conv_up_down_layers = None
    last_layer = None
    softmax_layer = None
    mc_dropout = None


    def __init__(self, *args, **kwargs):
        super().__init__(self._inputs, self._outputs, name='AlphaFold_Distance_Prediction_Model')
        self.dropout_mean = None
        print(kwargs)
        try:
            self.mc_sampling = kwargs["mc_sampling"]
        except:
            self.mc_sampling = 100

        if kwargs.get("class_weights", None) is None:
            self.cw = tf.ones((self.batch_size, self.crop_size, self.crop_size, self.output_channels))
        elif isinstance(kwargs.get("class_weights", None), dict):
            if self.output_channels != len(kwargs["class_weights"]):
                raise ValueError("Incorrect length of class weights. It should be equal to number of output channels!")
            # tf.tile(tf.reshape(tf.convert_to_tensor(list([d[i] for i in range(len(d))])), (1,1,1,5)),tf.constant([2,4,4,1]))
            self.cw = tf.convert_to_tensor(list([kwargs["class_weights"][i] for i in range(len(kwargs["class_weights"]))]), dtype=tf.float32)
        else:
            raise ValueError("Class weights must be a dictonary")
        # print(kwargs)

    def __new__(cls, input_channels, output_channels, num_blocks, num_channels, dilation, batch_size=64, crop_size=64,
                 non_linearity='elu', dropout_rate=0.0, reg_strength=1e-4, logits=True, sparse=False, kernel_initializer="he_normal",
                 kernel_regularizer="l2", mc_dropout=False, mc_sampling=100, class_weights=None):

        if (sum(num_blocks) % len(dilation)) != 0:
            raise ValueError('(Sum of ResNet block % Length of list containing dilation rates) == 0!')

        cls.input_channels = input_channels
        cls.output_channels = output_channels
        cls.crop_size = crop_size
        cls.num_blocks = num_blocks
        cls.num_channels = num_channels
        cls.dilation = dilation
        cls.batch_size = batch_size
        cls.non_linearity = non_linearity
        cls.dropout_rate = dropout_rate
        cls.reg_strength = reg_strength
        if kernel_initializer is None:
            cls.kernel_initializer = "glorot_uniform"    
        else:
            cls.kernel_initializer = kernel_initializer
        cls.logits = logits
        if kernel_regularizer=="l2":
            cls.kernel_regularizer = l2
        elif kernel_regularizer=="l1":
            cls.kernel_regularizer = l1
        elif kernel_regularizer=="l1_l2":
            cls.kernel_regularizer = l1_l2
        elif kernel_regularizer is None:
            cls.kernel_regularizer = cls.none_reg
        else:
            raise ValueError("Wrong type of regularizer selected!")
        cls.sparse = sparse
        # self.inputs = None
        cls.first_layers = None
        cls.resnet_stack = None
        cls.dropout_layers = None
        cls.identity_add_layer = None
        cls.conv_up_down_layers = None
        cls.last_layer = None
        cls.softmax_layer = None
        cls.mc_dropout = mc_dropout
        cls._inputs, cls._outputs = cls.model_init(cls.mc_dropout)
        return super().__new__(cls)

    @classmethod
    def model_init(cls, mcd=False):
        """Function that creates the network based on initialized
        parameters.
          Returns:
            Tensorflow Keras model.
        """
        # Create the input layer
        # Create the input layer

        inputs = x = Input(shape=(cls.crop_size, cls.crop_size, cls.input_channels), batch_size=cls.batch_size, name='input_crop')

        # Down- or up sample the input depending on number of input channels and number of channels in first RestNet
        # block
        first_layers = cls.make_layer(first='True')
        for layer in first_layers:
            x = layer(x)

        # Concatenate sets containing variable number of ResNet blocks
        for idx, num_set_blocks in enumerate(cls.num_blocks):
            for block_num in range(num_set_blocks):
                identity = x
                layers_resnet = cls.resnet_block(num_filters=cls.num_channels[idx], stride=1,
                                                  atou_rate=cls.dilation[block_num % 4], set_block=idx,
                                                  block_num=block_num, kernel_size=3)
                for layer in layers_resnet:
                    x = layer(x)
                x = Add(name='add_' + str(idx) + '_' + str(block_num))([x, identity])
                if 0.0 < cls.dropout_rate < 1.0 and ((idx is not len(cls.num_blocks)-1) or (block_num is not num_set_blocks-1)):
                    if mcd:
                        x = Dropout(rate=cls.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num))(x, training=True)
                    else:
                        x = Dropout(rate=cls.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num))(x)
                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(cls.num_blocks)):
                    if cls.num_channels[idx] > cls.num_channels[idx + 1]:
                        x = Conv2D(filters=cls.num_channels[idx + 1], kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                   data_format='channels_last', name='downscale_' + str(idx) + 'to' + str(idx + 1))(x)
                    elif cls.num_channels[idx] < cls.num_channels[idx + 1]:
                        x = Conv2DTranspose(filters=cls.num_channels[idx + 1], kernel_size=1, strides=1,
                                            kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                            data_format='channels_last',
                                            padding='same', name='upscale_' + str(idx) + 'to' + str(idx + 1))(x)
                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(cls.num_blocks)):
                    if cls.num_channels[idx] > cls.output_channels:
                        x = cls.make_layer()[0](x)
                    elif cls.num_channels[idx] < cls.output_channels:
                        x = cls.make_layer()[0](x)

        if cls.logits:
            out = Softmax(axis=3, name='softmax_layer')(x)
        else:
            out = x
        return inputs, out

    @classmethod
    def make_layer(cls, first='False'):
        """Generates a block of layers consisting of convolutional or transposed convolutional layers
        and BatchNorm layers.
          Args:
            first: Boolean to indicate whether this block of layers is before and or after the ResNet
                   blocks.
          Returns:
            List containing layer objects making up the layer block.
        """
        layers = []
        if first == 'True':
            if cls.input_channels > cls.num_channels[0]:
                layers.append(Conv2D(filters=cls.num_channels[0], kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                     data_format='channels_last', name='downscale_conv2d'))
            elif cls.input_channels < cls.num_channels[0]:
                layers.append(Conv2DTranspose(filters=cls.num_channels[0], kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                              data_format='channels_last', name='downscale_conv2dtranspose'))
            layers.append(BatchNormalization(name='downscale_bn'))
        elif first == 'False':
            if cls.num_channels[-1] < cls.output_channels:
                layers.append(Conv2DTranspose(filters=cls.output_channels, kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                              data_format='channels_last', name='upscale_conv2d'))
            elif cls.num_channels[-1] > cls.output_channels:
                layers.append(Conv2D(filters=cls.output_channels, kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                     data_format='channels_last', name='upscale_conv2d'))

        return layers

    def _reshape_mask_fn(self, y_true, sample_weight):
        new_shape = array_ops.shape(y_true)
        new_shape = new_shape[0:-1]
        mask = tf.reshape(sample_weight, shape=new_shape)
        return mask

    def train_step(self, data):
        x, y, sw = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # Reshape sample weights
            mask = self._reshape_mask_fn(y, sw)
            loss = self.compiled_loss(y*self.cw, y_pred*self.cw, mask)
            loss = tf.reduce_sum(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, mask)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y, sw = data
        mask = self._reshape_mask_fn(y, sw)
        # Compute predictions
        y_pred = self(x)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, mask)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred, mask)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def resnet_block(cls, num_filters, stride, atou_rate, set_block, block_num, kernel_size=3):
        """Generates a ResNet block.
          Args:
            num_filters: Number of channels in the input to this ResNet block.
            stride:      An integer or tuple/list of 3 integers, specifying the strides of the convolution
                         along each spatial dimension.
            atou_rate:   An integer or tuple/list of 3 integers, specifying the dilation rate to use for dilated
                         convolution. Can be a single integer to specify the same value for all spatial dimensions.
            set_block:   Integer indicating which set of ResNet blocks this ResNet block belongs to.
            block_num:   Integer indicating which position this ResNet block holds in its set of ResNet blockss.
            kernel_size: An integer or tuple/list of 3 integers, specifying the depth, height and width of the
                         3D convolution window. Can be a single integer to specify the same value for all spatial
                         dimensions.
          Returns:
            List containing the layer objects making up the ResNet block.
        """
        layers = []

        # Project down
        layers.append(BatchNormalization(name='batch_norm_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=cls.non_linearity,
                                 name='non_linearity_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=1, strides=stride, padding='same',
                             kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                             data_format='channels_last', name='conv_down_' + str(set_block) + '_' + str(block_num)))

        # Strided convolution
        layers.append(BatchNormalization(name='batch_norm_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=cls.non_linearity,
                                 name='non_linearity_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=kernel_size, strides=stride, padding='same',
                             dilation_rate=atou_rate, data_format='channels_last',
                             kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                             name='conv_dil_' + str(set_block) + '_' + str(block_num)))

        # Project up
        layers.append(BatchNormalization(name='batch_norm_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=cls.non_linearity,
                                 name='non_linearity_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2DTranspose(filters=num_filters, kernel_size=1, strides=stride, padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=cls.kernel_initializer, kernel_regularizer=cls.kernel_regularizer(cls.reg_strength),
                                      name='conv_up_' + str(set_block) + '_' + str(block_num)))

        return layers

    def mc_predict_with_sample(self, X):
        mc_predictions = []
        for _ in tqdm.tqdm(range(self.mc_sampling)):
            y_p = self.predict(X, batch_size=self.batch_size)
            mc_predictions.append(y_p)
        mc_predictions = tf.convert_to_tensor(mc_predictions, dtype=tf.float32)
        mean = tf.math.reduce_mean(mc_predictions, axis=0)
        
        # misspecification
        total_count = 0
        total_misspecification = 0
        for i in range(self.mc_sampling):
            sample_mis, count = sample_misspecification(mc_predictions[i], mean)
            total_count += count
            total_misspecification += sample_mis
        total_misspecification /= float(total_count)
        try:
            total_misspecification = tf.math.sqrt(tf.cast(total_misspecification))
        except:
            total_misspecification = math.sqrt(float(total_misspecification))
        return mc_predictions, mean, total_misspecification
    
    def mc_predict_without_sample(self, X):
        tmp_path = ".mcd_tmp_files"
        # start with mean computation
        mean = self.predict(X, batch_size=self.batch_size)
        save_tfrecord(mean, tmp_path + "/sample_"+str(0))
        for i in tqdm.tqdm(range(self.mc_sampling - 1)):
            pred = self.predict(X, batch_size=self.batch_size)
            save_tfrecord(pred, tmp_path + "/sample_"+str(i+1))
            mean += pred
        mean /= self.mc_sampling
        save_tfrecord(mean, tmp_path + "/mean")
        # misspecification
        total_count = 0
        total_misspecification = 0
        for i in range(self.mc_sampling):
            pred = read_tfrecord(tmp_path + "/sample_"+str(i))
            sample_mis, count = sample_misspecification(pred, mean)
            total_count += count
            total_misspecification += sample_mis
        total_misspecification /= float(total_count)
        try:
            total_misspecification = tf.math.sqrt(tf.cast(total_misspecification))
        except:
            total_misspecification = math.sqrt(float(total_misspecification))
        shutil.rmtree(tmp_path)
        return mean, total_misspecification

    def mc_predict(self, X, return_all=False):
        if return_all:
            return self.mc_predict_with_sample(X)
        else:
            out = self.mc_predict_without_sample(X)
            return None, out[0], out[1]
    