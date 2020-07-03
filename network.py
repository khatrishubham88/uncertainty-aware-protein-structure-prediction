from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Softmax
from trainable_model import CustomModel
from tensorflow.keras.regularizers import l2


class ResNet():
    """Two-dimensional dilated convolutional neural network with variable number of residual
    block groups. Each residual block group consists of four ResNet blocks.
    """
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, dilation, batch_size=64, crop_size=64,
                 non_linearity='elu', dropout_rate=0.0, reg_strength=1e-4, kernel_initializer="he_normal", logits=True):
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
        self.reg_strength = reg_strength
        self.kernel_initializer = kernel_initializer
        self.logits = logits

    def model(self):
        """Function that creates the network based on initialized
        parameters.
          Returns:
            Tensorflow Keras model.
        """
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
                x = Add(name='add_' + str(idx) + '_' + str(block_num))([x, identity])
                if 0.0 < self.dropout_rate < 1.0 and ((idx is not len(self.num_blocks)-1) or (block_num is not num_set_blocks-1)):
                    x = Dropout(rate=self.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num))(x)
                
                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(self.num_blocks)):
                    if self.num_channels[idx] > self.num_channels[idx + 1]:
                        x = Conv2D(filters=self.num_channels[idx + 1], kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(self.reg_strength),
                                   data_format='channels_last', name='downscale_' + str(idx) + 'to' + str(idx + 1))(x)
                    elif self.num_channels[idx] < self.num_channels[idx + 1]:
                        x = Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=l2(self.reg_strength),
                                            data_format='channels_last',
                                            padding='same', name='upscale_' + str(idx) + 'to' + str(idx + 1))(x)
                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(self.num_blocks)):
                    if self.num_channels[idx] > self.output_channels:
                        x = self.make_layer()[0](x)
                    elif self.num_channels[idx] < self.output_channels:
                        x = self.make_layer()[0](x)

        if self.logits:
            out = Softmax(axis=3, name='softmax_layer')(x)
        else:
            out = x
        distance_pred_resnet = CustomModel(inputs, out, name='AlphaFold_Distance_Prediction_Model')

        return distance_pred_resnet

    def make_layer(self, first='False'):
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
            if self.input_channels > self.num_channels[0]:
                layers.append(Conv2D(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=l2(self.reg_strength),
                                     data_format='channels_last', name='downscale_conv2d'))
            elif self.input_channels < self.num_channels[0]:
                layers.append(Conv2DTranspose(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=l2(self.reg_strength),
                                              data_format='channels_last', name='downscale_conv2dtranspose'))
            layers.append(BatchNormalization(name='downscale_bn'))
        elif first == 'False':
            if self.num_channels[-1] < self.output_channels:
                layers.append(Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=l2(self.reg_strength),
                                              data_format='channels_last', name='upscale_conv2d'))
            elif self.num_channels[-1] > self.output_channels:
                layers.append(Conv2D(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=l2(self.reg_strength),
                                     data_format='channels_last', name='upscale_conv2d'))

        return layers

    def resnet_block(self, num_filters, stride, atou_rate, set_block, block_num, kernel_size=3):
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
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_down_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=1, strides=stride, padding='same',
                             kernel_initializer=self.kernel_initializer, kernel_regularizer=l2(self.reg_strength),
                             data_format='channels_last', name='conv_down_' + str(set_block) + '_' + str(block_num)))

        # Strided convolution
        layers.append(BatchNormalization(name='batch_norm_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=kernel_size, strides=stride, padding='same',
                             dilation_rate=atou_rate, data_format='channels_last',
                             kernel_initializer=self.kernel_initializer,
                             kernel_regularizer=l2(self.reg_strength),
                             name='conv_dil_' + str(set_block) + '_' + str(block_num)))

        # Project up
        layers.append(BatchNormalization(name='batch_norm_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2DTranspose(filters=num_filters, kernel_size=1, strides=stride, padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=self.kernel_initializer,
                                      kernel_regularizer=l2(self.reg_strength),
                                      name='conv_up_' + str(set_block) + '_' + str(block_num)))

        return layers