import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

"""
To-Do List:
1. Test the model output on data from Kaggle.
2. Implement cross-entropy loss using masking matrix.
3. Weight initialization.
4. Look into position-specific bias mentioned in DeepMind paper.
"""


class ResNet:
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, dilation, batch_size=64, crop_size=64,
                 non_linearity='elu', dropout_rate=1.0):
        super(ResNet, self).__init__()
        if (sum(num_blocks) % len(dilation)) != 0:
            raise ValueError('(Sum of ResNet block % Length of list containing dilation rates) == 0!')

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.crop_size = crop_size
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.dilation = dilation
        self.batch_size = batch_size
        self.non_linearity = non_linearity
        self.dropout_rate = dropout_rate

    def model(self):
        # Create the input layer
        inputs = x = Input(shape=(self.crop_size, self.crop_size, self.input_channels), batch_size=self.batch_size,
                           name='input_crop')

        # Down- or up sample the input depending on number of input channels and number of channels in first RestNet
        # block
        first_layers = self.make_layer(first='True')
        for layer in first_layers:
            x = layer(x)

        # Concatenate sets containing variable number of ResNet blocks
        for idx, num_set_blocks in enumerate(self.num_blocks):
            for block_num in range(num_set_blocks):
                identity = x
                layers_resnet = self.resnet_block(num_filters=self.num_channels[idx], stride=1,
                                                  atou_rate=self.dilation[block_num % 4], set_block=idx,
                                                  block_num=block_num, kernel_size=3)
                for layer in layers_resnet:
                    x = layer(x)
                if self.dropout_rate < 1.0:
                    x = Dropout(rate=self.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num))(x)
                x = add([x, identity], name='add_' + str(idx) + '_' + str(block_num))

                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(self.num_blocks)):
                    if self.num_channels[idx] > self.num_channels[idx + 1]:
                        x = Conv2D(filters=self.num_channels[idx + 1], kernel_size=1, strides=1, padding='same',
                                   name='downscale_' + str(idx) + 'to' + str(idx + 1))(x)
                    elif self.num_channels[idx] < self.num_channels[idx + 1]:
                        x = Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                            padding='same', name='upscale_' + str(idx) + 'to' + str(idx + 1))(x)
                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(self.num_blocks)):
                    if self.num_channels[idx] > self.output_channels:
                        x = self.make_layer()[0](x)
                    elif self.num_channels[idx] < self.output_channels:
                        x = self.make_layer()[0](x)

        out = Softmax(axis=3, name='softmax_layer')(x)
        distance_pred_resnet = Model(inputs, out, name='AlphaFold_Distance_Prediction_Model')

        return distance_pred_resnet

    def make_layer(self, first='False'):
        layers = []
        if first == 'True':
            if self.input_channels > self.num_channels[0]:
                layers.append(Conv2D(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                     name='downscale_conv2d'))
            elif self.input_channels < self.num_channels[0]:
                layers.append(Conv2DTranspose(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                              name='downscale_conv2dtranspose'))
            layers.append(BatchNormalization(name='downscale_bn'))
        elif first == 'False':
            if self.num_channels[-1] < self.output_channels:
                layers.append(Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                              name='upscale_conv2d'))
            elif self.num_channels[-1] > self.output_channels:
                layers.append(Conv2D(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                     name='upscale_conv2d'))

        return layers

    def resnet_block(self, num_filters, stride, atou_rate, set_block, block_num, kernel_size=3):
        layers = []

        # Project down
        layers.append(BatchNormalization(name='batch_norm_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=1, strides=stride, padding='same',
                             name='conv_down_' + str(set_block) + '_' + str(block_num)))

        # Strided convolution
        layers.append(BatchNormalization(name='batch_norm_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=kernel_size, strides=stride, padding='same',
                             dilation_rate=atou_rate, name='conv_dil_' + str(set_block) + '_' + str(block_num)))

        # Project up
        layers.append(BatchNormalization(name='batch_norm_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2DTranspose(filters=num_filters, kernel_size=1, strides=stride, padding='same',
                                      name='conv_up_' + str(set_block) + '_' + str(block_num)))

        return layers


if __name__ == "__main__":
    #nn = ResNet(input_channels=64, output_channels=64, num_blocks=[4, 4], num_channels=[64, 32], dilation=[1, 2, 4, 8],
    #            batch_size=32, dropout_rate=0.15)
    #model = nn.model()

    #x = tf.keras.backend.random_normal(shape=(32, 64, 64, 64), mean=0.0, stddev=1.0)
    #y_pred = model(x)
    #y_true = K.softmax(tf.keras.backend.random_normal(shape=(32, 64, 64, 64), mean=0.0, stddev=1.0), axis=3)
    #mask = K.random_uniform(shape=(32, 64, 64), minval=0, maxval=2, dtype=tf.dtypes.int32)

    #loss = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    #l = loss(y_pred, y_true, mask)

    nn = ResNet(input_channels=2, output_channels=2, num_blocks=[4, 4], num_channels=[64, 32], dilation=[1, 2, 4, 8],
                batch_size=2, crop_size=2, dropout_rate=0.15)
    model = nn.model()

    x = tf.keras.backend.random_normal(shape=(2, 2, 2, 2), mean=0.0, stddev=1.0)
    y_pred = model(x)
    y_true = K.softmax(tf.keras.backend.random_normal(shape=(2, 2, 2, 2), mean=0.0, stddev=1.0), axis=3)
    mask = tf.convert_to_tensor(np.eye(2, 2))
    mask = K.reshape(mask, shape=(2, 2, 1))
    mask = K.tile(mask, (1, 1, 2))
    mask = tf.transpose(mask, perm=(2, 1, 0))

    """
    Testing based on 
    https://stackoverflow.com/questions/47057361/how-do-i-mask-a-loss-function-in-keras-with-the-tensorflow-backend
    
    It can be seen that both approaches yield the same loss matrix / loss sum.
    """

    loss = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    l = loss(y_pred, y_true) * K.cast_to_floatx(mask)
    print(K.sum(l)/(2*2*2))

    loss = CategoricalCrossentropy()
    l = loss(y_pred, y_true, mask)
    print(l)