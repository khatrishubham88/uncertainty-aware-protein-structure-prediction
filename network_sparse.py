from tensorflow.keras import Input
from tensorflow import keras
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, Softmax
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.python.ops import array_ops
import tensorflow as tf
import time
import tqdm
from utils import output_to_distancemaps


class ResNet(keras.Model):
    """Two-dimensional dilated convolutional neural network with variable number of residual
    block groups. Each residual block group consists of four ResNet blocks.
    """
    def __init__(self, input_channels, output_channels, num_blocks, num_channels, dilation, batch_size=64, crop_size=64,
                 non_linearity='elu', dropout_rate=0.0, reg_strength=1e-4, logits=True, sparse=False, kernel_initializer="he_normal",
                 kernel_regularizer="l2"):
        super(ResNet, self).__init__(name='AlphaFold_Distance_Prediction_Model')
        
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
        if kernel_regularizer=="l2":
            self.kernel_regularizer = l2
        elif kernel_regularizer=="l1":
            self.kernel_regularizer = l1
        elif kernel_regularizer=="l1_l2":
            self.kernel_regularizer = l1_l2
        else:
            raise ValueError("Wrong type of regularizer selected!")
        self.sparse = sparse
        # self.inputs = None
        self.first_layers = None
        self.resnet_stack = None
        self.dropout_layers = None
        self.identity_add_layer = None
        self.conv_up_down_layers = None
        self.last_layer = None
        self.softmax_layer = None
        self.model_init()
        print("\nFirst Layer: \n {}\n".format(self.first_layers))
        print("\nresnet_stack Layer: \n {}\n".format(self.resnet_stack))
        print("\nidentity_add_layer: \n {}\n".format(self.identity_add_layer))
        print("\ndropout_layers: \n {}\n".format(self.dropout_layers))
        print("\nconv_up_down_layers: \n {}\n".format(self.conv_up_down_layers))
        print("\nlast_layer: \n {}\n".format(self.last_layer))
        print("\nsoftmax_layer: \n {}\n".format(self.softmax_layer))
        
        # time.sleep(120)
        
        self.build((self.batch_size, self.crop_size, self.crop_size, self.input_channels))

    def model_init(self):
        """Function that creates the network based on initialized
        parameters.
          Returns:
            Tensorflow Keras model.
        """
        # Create the input layer
        # self.inputs = Input(shape=(self.crop_size, self.crop_size, self.input_channels), batch_size=self.batch_size,
        #                    name='input_crop')

        # Down- or up sample the input depending on number of input channels and number of channels in first RestNet
        # block
        self.first_layers = self.make_layer(first='True')

        self.resnet_stack = []
        self.dropout_layers = []
        self.conv_up_down_layers = []
        self.identity_add_layer = []
        # self.conv_down_layers = []
        
        # Concatenate sets containing variable number of ResNet blocks
        for idx, num_set_blocks in enumerate(self.num_blocks):
            resnet_block_list = []
            block_dropout_list = []
            block_channel_adjustment = []
            block_identity_add = []
            for block_num in range(num_set_blocks):
                # identity = x
                layers_resnet = self.resnet_block(num_filters=self.num_channels[idx], stride=1,
                                                  atou_rate=self.dilation[block_num % 4], set_block=idx,
                                                  block_num=block_num, kernel_size=3)
                resnet_block_list.append(layers_resnet)
                # for layer in layers_resnet:
                #     x = layer(x)
                # x = Add(name='add_' + str(idx) + '_' + str(block_num))([x, identity])
                block_identity_add.append(Add(name='add_' + str(idx) + '_' + str(block_num)))
                
                if 0.0 < self.dropout_rate < 1.0 and ((idx is not len(self.num_blocks)-1) or (block_num is not num_set_blocks-1)):
                    # x = Dropout(rate=self.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num))(x)
                    block_dropout_list.append(Dropout(rate=self.dropout_rate, name='dropout_' + str(idx) + '_' + str(block_num)))
                
                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(self.num_blocks)):
                    if self.num_channels[idx] > self.num_channels[idx + 1]:
                        # x = Conv2D(filters=self.num_channels[idx + 1], kernel_size=1, strides=1, padding='same',
                        #            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                        #            data_format='channels_last', name='downscale_' + str(idx) + 'to' + str(idx + 1))(x)
                        block_channel_adjustment.append(Conv2D(filters=self.num_channels[idx + 1], kernel_size=1, strides=1, padding='same',
                                   kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                   data_format='channels_last', name='downscale_' + str(idx) + 'to' + str(idx + 1)))
                        
                        
                    elif self.num_channels[idx] < self.num_channels[idx + 1]:
                        # x = Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                        #                     kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                        #                     data_format='channels_last',
                        #                     padding='same', name='upscale_' + str(idx) + 'to' + str(idx + 1))(x)
                        block_channel_adjustment.append(Conv2DTranspose(filters=self.num_channels[idx + 1], kernel_size=1, strides=1,
                                            kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                            data_format='channels_last',
                                            padding='same', name='upscale_' + str(idx) + 'to' + str(idx + 1)))
                        
                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(self.num_blocks)):
                    if self.num_channels[idx] > self.output_channels:
                        # x = self.make_layer()[0](x)
                        self.last_layer = self.make_layer()[0]
                    elif self.num_channels[idx] < self.output_channels:
                        # x = self.make_layer()[0](x)
                        self.last_layer = self.make_layer()[0]
            self.resnet_stack.append(resnet_block_list)
            self.dropout_layers.append(block_dropout_list)
            self.conv_up_down_layers.append(block_channel_adjustment)
            self.identity_add_layer.append(block_identity_add)    

        if self.logits:
            # out = Softmax(axis=3, name='softmax_layer')(x)
            self.softmax_layer = Softmax(axis=3, name='softmax_layer')
        # else:
        #     out = x
        # distance_pred_resnet = CustomModel(inputs, out, sparse=self.sparse)

        # return distance_pred_resnet

    def call(self, inputs, training=False):
        output = inputs
        for layer in self.first_layers:
            output = layer(output)
        tf.print("\n\nFirst layer done!\n\n")
        print("\n\nFirst layer done!\n\n")
        
        for idx, num_set_blocks in enumerate(self.num_blocks):
            block_resnet_stack_idx = 0
            block_dropout_layers_idx = 0
            block_conv_up_down_layers_idx = 0
            for block_num in range(num_set_blocks):
                identity = output
                tf.print("\n\n Doing pass for Resnet: \n {}".format(self.resnet_stack[idx][block_resnet_stack_idx]))
                print("\n\n Doing pass for Resnet: \n {}".format(self.resnet_stack[idx][block_resnet_stack_idx]))
                for layer in self.resnet_stack[idx][block_resnet_stack_idx]:
                    output = layer(output)
                tf.print("\n\n Doing pass for Add: \n {}".format(self.identity_add_layer[idx][block_resnet_stack_idx]))
                print("\n\n Doing pass for Add: \n {}".format(self.identity_add_layer[idx][block_resnet_stack_idx]))
                output = self.identity_add_layer[idx][block_resnet_stack_idx]([output, identity])
                block_resnet_stack_idx += 1
                if 0.0 < self.dropout_rate < 1.0 and ((idx is not len(self.num_blocks)-1) or (block_num is not num_set_blocks-1)):
                    output = self.dropout_layers[idx][block_dropout_layers_idx](output, training)
                    tf.print("\n\n Doing pass for Dropout: \n {}".format(self.dropout_layers[idx][block_dropout_layers_idx]))
                    print("\n\n Doing pass for Dropout: \n {}".format(self.dropout_layers[idx][block_dropout_layers_idx]))
                    block_dropout_layers_idx += 1
                    
                if ((block_num + 1) == num_set_blocks) and ((idx + 1) != len(self.num_blocks)):
                    if (self.num_channels[idx] > self.num_channels[idx + 1]) or (self.num_channels[idx] < self.num_channels[idx + 1]):
                        output = self.conv_up_down_layers[idx][block_conv_up_down_layers_idx](output)
                        block_conv_up_down_layers_idx += 1
                elif ((block_num + 1) == num_set_blocks) and ((idx + 1) == len(self.num_blocks)):
                    if (self.num_channels[idx] > self.output_channels) or (self.num_channels[idx] < self.output_channels):
                        output = self.last_layer(output)
                
        if self.logits:
            output = self.softmax_layer(output)
        return output
        
    
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
                                     kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                     data_format='channels_last', name='downscale_conv2d'))
            elif self.input_channels < self.num_channels[0]:
                layers.append(Conv2DTranspose(filters=self.num_channels[0], kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                              data_format='channels_last', name='downscale_conv2dtranspose'))
            layers.append(BatchNormalization(name='downscale_bn'))
        elif first == 'False':
            if self.num_channels[-1] < self.output_channels:
                layers.append(Conv2DTranspose(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                              kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                              data_format='channels_last', name='upscale_conv2d'))
            elif self.num_channels[-1] > self.output_channels:
                layers.append(Conv2D(filters=self.output_channels, kernel_size=1, strides=1, padding='same',
                                     kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                     data_format='channels_last', name='upscale_conv2d'))

        return layers
    
    def _reshape_mask_fn(self, y_true, sample_weight):
        new_shape = array_ops.shape(y_true)
        new_shape = new_shape[0:-1]
        mask = tf.reshape(sample_weight, shape=new_shape)
        return mask
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sw = data
        # tf.print(self.layers[0].input_shape)
        # print("Printing from the train step")
        # proto_seq = tf.make_tensor_proto(sw)
        # numpy_seq = tf.make_ndarray(proto_seq)
        # sh = array_ops.shape(x)
        # tf.print(sh)
        #  print(x.numpy()[0])
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # Reshape sample weights
            # new_shape = array_ops.shape(y)
            
            # tf.print("\n\ny shape = {}, mask shape = {}\n\n".format(new_shape.numpy(), array_ops.shape(sw).numpy()))
            # new_shape = new_shape[0:-1]
            # tf.print(new_shape)
            # mask = tf.reshape(sw, shape=self._get_mask_shape_fn(y))
            mask = self._reshape_mask_fn(y, sw)
            # mask = K.variable(mask)
            # test_loss = self.compiled_loss(y, y_pred)
            # test_loss = test_loss*mask
            loss = self.compiled_loss(y, y_pred, mask)
            # num_mismatch = 0
            # for i in range(array_ops.shape(test_loss)[0]):
            #     for j in range(array_ops.shape(test_loss)[1]):
            #         for k in range(array_ops.shape(test_loss)[2]):
            #             if tf.math.equal(test_loss, loss).numpy()[i,j,k]:
            #                 pass
            #             elif not tf.math.equal(test_loss, loss).numpy()[i,j,k]:
            #                 num_mismatch +=1 
            # # print("Total mismatch = {}".format(num_mismatch))
                            
            # tf.print("test_loss shape = {}, loss shape = {}".format(array_ops.shape(test_loss), array_ops.shape(loss)))
            # loss = tf.reduce_sum(self.compiled_loss(y, y_pred)*mask*(1./)
            # loss = loss*mask
            # loss = K.sum(K.sum(K.sum(loss)))
            # tf.print("\n\nloss = {}, factor = {}\n\n".format(tf.reduce_sum(loss).numpy(), (tf.constant(1.)/tf.cast(new_shape[0], dtype=tf.float32)).numpy()))
            loss = tf.reduce_sum(loss)
            
            
        # print("Forward pass went through")
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred, mask)
        # tf.print("\n\nCompiled metric = {}\n\n".format(self.compiled_metrics))
        # tf.print("\n\nTraining Output = {}\n\n".format({m.name: m.result() for m in self.metrics}))
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        # Unpack the data
        x, y, sw = data
        mask = self._reshape_mask_fn(y, sw)
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, mask)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred, mask)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

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
                             kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                             data_format='channels_last', name='conv_down_' + str(set_block) + '_' + str(block_num)))

        # Strided convolution
        layers.append(BatchNormalization(name='batch_norm_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_conv_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2D(filters=num_filters//2, kernel_size=kernel_size, strides=stride, padding='same',
                             dilation_rate=atou_rate, data_format='channels_last',
                             kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                             name='conv_dil_' + str(set_block) + '_' + str(block_num)))

        # Project up
        layers.append(BatchNormalization(name='batch_norm_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Activation(activation=self.non_linearity,
                                 name='non_linearity_up_' + str(set_block) + '_' + str(block_num)))
        layers.append(Conv2DTranspose(filters=num_filters, kernel_size=1, strides=stride, padding='same',
                                      data_format='channels_last',
                                      kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer(self.reg_strength),
                                      name='conv_up_' + str(set_block) + '_' + str(block_num)))

        return layers
    
    
class ResNetV2(keras.Model):
    """Two-dimensional dilated convolutional neural network with variable number of residual
    block groups. Each residual block group consists of four ResNet blocks.
    """
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
        cls.kernel_initializer = kernel_initializer
        cls.logits = logits
        if kernel_regularizer=="l2":
            cls.kernel_regularizer = l2
        elif kernel_regularizer=="l1":
            cls.kernel_regularizer = l1
        elif kernel_regularizer=="l1_l2":
            cls.kernel_regularizer = l1_l2
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
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sw = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # Reshape sample weights
            mask = self._reshape_mask_fn(y, sw)
            # cw_y_true = tf.math.multiply(y, self.cw)
            # cw_y_pred = tf.math.multiply(y_pred, self.cw)
            loss = self.compiled_loss(y*self.cw, y_pred*self.cw, mask)
            
            loss = tf.reduce_sum(loss)
           # del cw_y_pred
           # del cw_y_true
            
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
    
    def mc_predict(self, X, min_val, max_val, num_bin):
        mc_predictions = []
        for _ in tqdm.tqdm(range(self.mc_sampling)):
            y_p = self.predict(X)
            mc_predictions.append(y_p)
        # mean = tf.math.reduce_mean(tf.convert_to_tensor(mc_predictions, dtype=tf.float32), axis
        mc_predictions = tf.convert_to_tensor(mc_predictions, dtype=tf.float32)
        mean = tf.math.reduce_mean(mc_predictions, axis=0)
        # print(mean.shape)
        return mc_predictions, mean
