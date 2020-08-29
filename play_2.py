import tensorflow as tf
import tensorflow.keras.layers as layers
import traceback
import contextlib

@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
"""
Predefine all necessary layer for the R-CNN
"""
class Conv_bn_relu(tf.keras.layers.Layer):
    def __init__(self, filter_height, filter_width, num_filters, stride_y, stride_x, name = None, is_training=True, bias=False, padding='SAME'):
        super(Conv_bn_relu, self).__init__(name= name)
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.num_filters = num_filters
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.is_training = is_training
        self.is_bias = bias
        self.padding= padding
        self.bn = tf.keras.layers.BatchNormalization(center=True, scale=True, trainable = is_training)
    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_size = (self.filter_height, self.filter_width)
        self.conv2d = tf.keras.layers.Conv2D(self.num_filters, kernel_size=kernel_size, padding= self.padding,
                                             strides= (self.stride_y, self.stride_x), use_bias= self.is_bias,
                                             kernel_initializer="random_normal", bias_initializer= "zeros")

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x

class Conv(tf.keras.layers.Layer):
    def __init__(self, filter_height, filter_width, num_filters, stride_y, stride_x, name=None, padding='SAME'):
        super(Conv, self).__init__(name= name)
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.num_filters = num_filters
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.padding = padding
    def build(self, input_shape):
        kernel_size = (self.filter_height, self.filter_width)
        self.conv2d = tf.keras.layers.Conv2D(self.num_filters, kernel_size, (self.stride_y, self.stride_x),
                                             padding=self.padding, kernel_initializer="random_normal",
                                             use_bias= True, bias_initializer="zeros")
    def call(self, inputs):
        x = self.conv2d(inputs)
        x = tf.nn.relu(x)
        return x

class Fc(tf.keras.layers.Layer):
    def __init__(self, num_outs, name = None, relu=True):
        super(Fc, self).__init__(name=name)
        self.num_outs = num_outs
        self.is_relu = relu
        if relu:
            activation = "relu"
        else:
            activation = None
        self.fc = tf.keras.layers.Dense(units= num_outs, activation= activation)
    def call(self, inputs):
        x = self.fc(inputs)
        return x

#3, 3, 64, 1, 1, is_training=self.istraining, padding='SAME', name='conv1'
import os
import numpy as np
path = os.path.abspath('fold1_room1_mix001_ov1.npy')
'''layer = Conv(3,3,64,1,1, padding='SAME', name= 'conv')
x = np.load(path)
x = tf.constant(x, shape=[10, 300, 64, 7])
'''
layer = Fc(num_outs= 10, name="fc", relu= True)
x = tf.random.normal()

with tf.GradientTape(persistent=True) as tape:
    loss = layer(x)
    grads = tape.gradient(loss, layer.trainable_variables)
print([var.name for var in tape.watched_variables()])
print(len(grads))