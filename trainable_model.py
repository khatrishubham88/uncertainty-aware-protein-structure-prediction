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
        
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y, sw = data
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
            loss = self.compiled_loss(y, y_pred)
            # Reshape sample weights
            new_shape = array_ops.shape(y)
            tf.print(new_shape)
            new_shape = new_shape[0:-1]
            tf.print(new_shape)
            mask = tf.reshape(sw, shape=new_shape)
            # mask = K.variable(mask)
            loss = loss*mask
            loss = K.sum(K.sum(K.sum(loss)))
            
        # print("Forward pass went through")
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
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
    for i in dataprovider.take(10):
        print (i)
    print(dataprovider)
    K.clear_session()
    nn = ResNet(input_channels=20, output_channels=64, num_blocks=[28], num_channels=[64], dilation=[1, 2, 4, 8],
                batch_size=16, crop_size=64, dropout_rate=0.15)
    model = nn.model()
    
    model = CustomModel(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001),
                  loss=masked_categorical_cross_entropy_test())
    
    model.fit(dataprovider, epochs=3)
    