import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Input, Softmax


class ResNetBlock(Model):
    def __init__(self, num_filters, non_linearity, stride, padding, atou_rate, set_block, block_num, kernel_size=3):
        super(ResNetBlock, self).__init__(name='')
        self.num_filters = num_filters
        self.non_linearity = non_linearity
        self.stride = stride
        self.padding = padding
        self.atou_rate = atou_rate
        self.set_block = set_block
        self.block_num = block_num
        self.kernel_size = kernel_size

        self.bn_down = BatchNormalization()
        self.non_linearity_down = Activation(activation=self.non_linearity)
        self.conv_down = Conv2D(filters=self.num_filters//2, kernel_size=1, strides=self.stride, padding=self.padding,
                                data_format='channels_last', name='conv_down_' + str(set_block) + '_' + str(block_num))

        self.bn_conv = BatchNormalization()
        self.non_linearity_conv = Activation(activation=self.non_linearity)
        self.conv_dil = Conv2D(filters=self.num_filters//2, kernel_size=self.kernel_size, strides=self.stride,
                               padding=self.padding, dilation_rate=self.atou_rate, data_format='channels_last',
                               name='conv_dil_' + str(set_block) + '_' + str(block_num))

        self.bn_up = BatchNormalization()
        self.non_linearity_up = Activation(activation=self.non_linearity)
        self.conv_up = Conv2DTranspose(filters=self.num_filters, kernel_size=1, strides=self.stride,
                                       padding=self.padding, data_format='channels_last',
                                       name='conv_up_' + str(set_block) + '_' + str(block_num))

    def call(self, x, training=None):
        identity = x

        x = self.bn_down(x)
        x = self.non_linearity_down(x)
        x = self.conv_down(x)

        x = self.bn_conv(x)
        x = self.non_linearity_conv(x)
        x = self.conv_dil(x)

        x = self.bn_up(x)
        x = self.non_linearity_up(x)
        x = self.conv_up(x)

        out = identity + x

        return out


class ResNet(Model):
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, dilation, learning_rate,
                 kernel_size=3, batch_size=64, crop_size=64, non_linearity='elu', padding='same', dropout_rate=1.0):
        super(ResNet, self).__init__(name='')
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.dilation = dilation
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.non_linearity = non_linearity
        self.padding = padding
        self.dropout_rate = dropout_rate

        if (sum(self.num_blocks) % len(self.dilation)) != 0:
            raise ValueError('(Sum of ResNet block % Length of list containing dilation rates) == 0!')

        # Input Layer
        self.input_layer = Input(shape=(self.crop_size, self.crop_size, self.input_channels),
                                 batch_size=self.batch_size, name='input_crop')

        # Input convolutional and batch normalization layer
        if self.input_channels > self.num_channels[0]:
            self.conv_scale_in = Conv2D(filters=self.num_channels[0], kernel_size=1, strides=1, padding=self.padding,
                                        data_format='channels_last', name='downscale_conv2d')
        elif self.input_channels < self.num_channels[0]:
            self.conv_scale_in = Conv2DTranspose(filters=self.num_channels[0], kernel_size=1, strides=1,
                                                 padding=self.padding, data_format='channels_last',
                                                 name='downscale_conv2dtranspose')
        self.bn_scale_in = BatchNormalization()

        # ResNet blocks
        self.blocks = []
        for idx, num_set_blocks in enumerate(self.num_blocks):
            for block_num in range(num_set_blocks):
                self.blocks.append(ResNetBlock(num_filters=self.num_channels[idx], non_linearity=self.non_linearity,
                                               stride=1, padding=self.padding, atou_rate=self.dilation[block_num % 4],
                                               set_block=idx, block_num=block_num, kernel_size=self.kernel_size))
                if self.dropout_rate < 1.0:
                    self.blocks.append(Dropout(rate=self.dropout_rate))

                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(self.num_blocks)):
                    if self.num_channels[idx] > self.num_channels[idx + 1]:
                        self.blocks.append(Conv2D(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                                  padding=self.padding, data_format='channels_last',
                                                  name='downscale_' + str(idx) + 'to' + str(idx + 1)))
                    elif self.num_channels[idx] < self.num_channels[idx + 1]:
                        self.blocks.append(Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                                           data_format='channels_last', padding=self.padding,
                                                           name='upscale_' + str(idx) + 'to' + str(idx + 1)))

                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(self.num_blocks)):
                    if self.num_channels[idx] > self.output_channels:
                        self.blocks.append(Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1,
                                                           padding=self.padding, data_format='channels_last',
                                                           name='upscale_conv2d'))
                    elif self.num_channels[idx] < self.output_channels:
                        self.blocks.append(Conv2D(filters=self.output_channels, kernel_size=1, strides=1,
                                                  padding=self.padding, data_format='channels_last',
                                                  name='upscale_conv2d'))

        # Output Layer
        self.output_layer = Softmax(axis=3, name='softmax_output_layer')

    def call(self, x, training=None):
        x = self.conv_scale_in(x)
        x = self.bn_scale_in(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        out = self.output_layer(x)
        
        return out