from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Softmax


"""
To-Do List:
1. Test the model output on data from Kaggle.
2. Weight initialization.
3. Look into position-specific bias mentioned in DeepMind paper.
"""


class ResNet:
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, dilation, callbacks, learning_rate,
                 batch_size=64, crop_size=64, non_linearity='elu', dropout_rate=1.0,):
        super(ResNet, self).__init__()
        if (sum(num_blocks) % len(dilation)) != 0:
            raise ValueError('(Sum of ResNet block % Length of list containing dilation rates) == 0!')

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.crop_size = crop_size
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.dilation = dilation
        self.callbacks = callbacks
        self.learning_rate = learning_rate
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
                                   data_format='channels_last', name='downscale_' + str(idx) + 'to' + str(idx + 1))(x)
                    elif self.num_channels[idx] < self.num_channels[idx + 1]:
                        x = Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                            data_format='channels_last',
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
                                     data_format='channels_last', name='downscale_conv2d'))
            elif self.input_channels < self.num_channels[0]:
                layers.append(Conv2DTranspose(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                              data_format='channels_last', name='downscale_conv2dtranspose'))
            layers.append(BatchNormalization(name='downscale_bn'))
        elif first == 'False':
            if self.num_channels[-1] < self.output_channels:
                layers.append(Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                              data_format='channels_last', name='upscale_conv2d'))
            elif self.num_channels[-1] > self.output_channels:
                layers.append(Conv2D(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                     data_format='channels_last', name='upscale_conv2d'))

        return layers

    def resnet_block(self, num_filters, stride, atou_rate, set_block, block_num, kernel_size=3):
        layers = []

        # Project down
        layers.append(BatchNormalization(name='batch_norm_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=1, strides=stride, padding='same',
                             data_format='channels_last', name='conv_down_' + str(set_block) + '_' + str(block_num)))

        # Strided convolution
        layers.append(BatchNormalization(name='batch_norm_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=kernel_size, strides=stride, padding='same',
                             dilation_rate=atou_rate, data_format='channels_last',
                             name='conv_dil_' + str(set_block) + '_' + str(block_num)))

        # Project up
        layers.append(BatchNormalization(name='batch_norm_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2DTranspose(filters=num_filters, kernel_size=1, strides=stride, padding='same',
                                      data_format='channels_last',
                                      name='conv_up_' + str(set_block) + '_' + str(block_num)))

        return layers