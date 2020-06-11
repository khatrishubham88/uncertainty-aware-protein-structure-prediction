import tensorflow as tf
from tensorflow import keras
# from network import ResNet
import tensorflow.keras.backend as K
from utils import masked_categorical_cross_entropy_test
import numpy as np
from dataprovider import DataGenerator
from tensorflow.python.ops import array_ops


class CustomModel(keras.Model):
    # def __init__(self, input_dim, output_dim):
    #     super.__init__()
    #     self.input_dim = input_dim
    #     self.output_dim = output_dim
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

if __name__=="__main__":
    
    # get data
    path = ["../proteinnet/data/casp7/training/100/1"]
    params = {
    "dim": (64,64), # this is the LxL
    "datasize": None, 
    "features": "primary", # this will decide the number of channel, with primary 20, secondary 20+something
    "padding_value": -1, # value to use for padding the sequences, mask is padded by 0 only
    "minimum_bin_val": 2, # starting bin size
    "maximum_bin_val": 22, # largest bin size
    "num_bins": 64,         # num of bins to use
    "batch_size": 32,       # batch size for training, check if this is needed here or should be done directly in fit?
    "shuffle": False,        # if wanna shuffle the data, this is not necessary
    "shuffle_buffer_size": None,     # if shuffle is on size of shuffle buffer, if None then =batch_size
    "random_crop": False,         # if cropping should be random, this has to be implemented later
    "flattening" : True,        # if flattten the mask
    # "take" : 128,
    }
    dataprovider = DataGenerator(path, **params).datafeeder
    # for i in dataprovider.take(10):
    #     print (i)
    # print(dataprovider)
    K.clear_session()
    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=params.batch_size, crop_size=params.dim[0], dropout_rate=0.15)
    model = nn.model()
    
    model = CustomModel(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001),
                  loss=masked_categorical_cross_entropy_test())
    
    model.fit(dataprovider, epochs=3)
    