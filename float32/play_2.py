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
        self.bn = tf.keras.layers.BatchNormalization(center=True, scale=True, trainable = True)
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

class Max_pool(tf.keras.layers.Layer):
    def __init__(self, filter_height, filter_width, stride_y, stride_x, name=None, padding='SAME'):
        super(Max_pool, self).__init__(name=name)
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.padding = padding
        self.max_pool2d = tf.keras.layers.MaxPool2D((filter_height, filter_width), strides=(stride_y, stride_x),
                                                    padding= padding)
    def call(self, inputs):
        return self.max_pool2d(inputs)

class Avr_pool(tf.keras.layers.Layer):
    def __init__(self, filter_height, filter_width, stride_y, stride_x, name=None, padding='SAME'):
        super(Avr_pool, self).__init__(name=name)
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.padding = padding
        self.avr_pool = tf.keras.layers.AvgPool2D((filter_height, filter_width), (stride_y, stride_x), padding=padding)
    def call(self, inputs):
        return self.avr_pool(inputs)

class Dropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = rate
        self.drop = tf.keras.layers.Dropout(rate=rate)
    def call(self, inputs, training):
        return self.drop(inputs, training)

class GRUs(tf.keras.layers.Layer):
    def __init__(self, units, num_cells = 1, input_keep_prob=1.0, output_keep_prob=1.0, go_backwards =False,
                 return_sequences = True, return_state = False, name = None, *args, **kwargs):
        super(GRUs, self).__init__(name=name)
        self.num_cells = num_cells
        self.dim = units #nhidden = 256
        self.input_drop = abs(1 -input_keep_prob)
        self.output_drop = abs(1 -output_keep_prob)
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.states = None
        def gruCell():
            return tf.keras.layers.GRU(units, dropout= self.input_drop, go_backwards= self.go_backwards,
                                       return_state=True, return_sequences=True, stateful=False)
        self._layers_name = ['GruCell_' +str(i) for i in range(num_cells)]
        for name in self._layers_name:
            self.__setattr__(name, gruCell())
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_cells': self.num_cells,
            'input_drop': self.input_drop,
            'output_drop': self.output_drop,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            '_layers_name': self._layers_name,
            'units': self.dim
        })
        return config

    def call(self, inputs):
        seq = inputs
        state = None
        for name in self._layers_name:
            cell = self.__getattribute__(name)
            (seq, state) = cell(seq, initial_state = state)
        self.states = state
        if self.return_state:
            if self.return_sequences:
                return [seq, state]
            return [seq[:,-1,:], state]
        if self.return_sequences:
            return seq
        return seq[:, -1, :]

class Bidirectional(tf.keras.layers.Layer):
    def __init__(self, nhidden, nlayer, input_keep_prob=1.0, output_keep_prob=1.0, return_state=False,
                return_sequences = True, name = None, *args, **kwargs):
        super(Bidirectional, self).__init__(name = name, *args, **kwargs)
        self.num_hiddens = nhidden
        self.num_cells = nlayer
        self.input_drop = abs(1 - input_keep_prob)
        self.output_drop = abs(1- output_keep_prob)
        self.fw_cells = GRUs(nhidden, nlayer, input_keep_prob, output_keep_prob,
                             go_backwards=False, return_sequences= return_sequences, return_state= return_state)
        self.bw_cells = GRUs(nhidden, nlayer, input_keep_prob, output_keep_prob,
                             go_backwards=True, return_sequences= return_sequences, return_state= return_state)
        self.return_state = return_state
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.biGrus = tf.keras.layers.Bidirectional(self.fw_cells, backward_layer=self.bw_cells, input_shape=input_shape)

    def call(self, inputs):
        out =  self.biGrus(inputs)
        if self.return_state:
            states = tf.keras.layers.concatenate((out[1], out[2]), axis=1)
            return out[0], states
        return out

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attention_size, scaled_=True, masked_=False, name="self-attention", *args, **kwargs):
        super(SelfAttention, self).__init__(name= name, *args, **kwargs)
        self.attention_size = attention_size
        self.scaled = scaled_
        self.masked = masked_
        self.attention = None

    def build(self, input_shape): #(batch_num, seq_len, ndim)
        ndim = input_shape[2]
        self.Q = tf.keras.layers.Dense(self.attention_size)
        self.K = tf.keras.layers.Dense(self.attention_size)
        self.V = tf.keras.layers.Dense(ndim)

    def call(self, inputs):
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        self.attention = tf.matmul(q,k,transpose_b=True)
        if self.scaled:
            d_k = tf.cast(k.shape[-1], dtype = tf.float32)
            self.attention = tf.divide(self.attention, tf.sqrt(d_k))
        if self.masked:
            raise  NotImplementedError
        self.attention = tf.nn.softmax(self.attention, axis= -1)
        return tf.matmul(self.attention, v)

#3, 3, 64, 1, 1, is_training=self.istraining, padding='SAME', name='conv1'
import os
import numpy as np
'''
x = tf.constant(x, shape=[10, 300, 448])

x = tf.random.normal(shape=[32, 5, 10])
layer = SelfAttention(64)
print(layer(x).shape)
with tf.GradientTape(persistent=True) as tape:
    loss = layer(x)
    grads = tape.gradient(loss, layer.trainable_variables)
for var in tape.watched_variables():
    print(var.name)
print(len(grads))'''

class CRNN(tf.keras.layers.Layer):
    def __init__(self, is_training, params, name, out_shape_sed, out_shape_doa,
                 dropout_keep_prob_cnn=None, dropout_keep_prob_rnn=None, dropout_keep_prob_dnn=None, *args, **kwargs):
        super(CRNN, self).__init__(name=name, *args, **kwargs)
        self.dropout_keep_prob_cnn = dropout_keep_prob_cnn
        if dropout_keep_prob_cnn is None:
            self.dropout_keep_prob_cnn= params['dropout_keep_prob_cnn']
        self.dropout_keep_prob_rnn = dropout_keep_prob_rnn
        if dropout_keep_prob_rnn is None:
            self.dropout_keep_prob_rnn = params['dropout_keep_prob_rnn']
        self.dropout_keep_prob_dnn = dropout_keep_prob_dnn
        if dropout_keep_prob_dnn is None:
            self.dropout_keep_prob_dnn = params['dropout_keep_prob_dnn']
        self.is_training = is_training
        self.params = params
        self.out_shape_sed = out_shape_sed
        self.out_shape_doa = out_shape_doa
        # (-1, 300, 64, 7)
        self.conv1 = Conv_bn_relu(filter_height= 3, filter_width= 3, num_filters= 64, stride_y= 1, stride_x= 1,
                                  name='conv1', is_training= is_training, padding='SAME')

        # (-1, 300, 64, 64)
        self.conv2 = Conv_bn_relu(filter_height= 3, filter_width= 3, num_filters= 64, stride_y= 1, stride_x= 1,
                                  name='conv2', is_training= is_training, padding='SAME')
        # (-1, 300, 64, 64)
        self.max_pool2 = Max_pool(filter_height= 5, filter_width=2, stride_y=5, stride_x= 2, name='max_pool2', padding='VALID')
        # (-1, 60, 32, 64)
        self.drop2 = Dropout(self.dropout_keep_prob_cnn)

        # (-1, 60, 32, 64)
        self.conv3 = Conv_bn_relu(filter_height=3,filter_width= 3, num_filters= 128,stride_y= 1, stride_x= 1,
                                  is_training=is_training, padding='SAME', name='conv3')
        # (-1, 60, 32, 128)
        self.max_pool3 = Max_pool(filter_height=1, filter_width= 2, stride_y=1, stride_x= 2, padding='VALID', name='max_pool3')
        # (-1, 60, 16, 128)
        self.drop3 = Dropout(self.dropout_keep_prob_cnn)

        # (-1, 60, 16, 128)
        self.conv4 = Conv_bn_relu(filter_height=3, filter_width=3, num_filters=128, stride_y=1, stride_x=1,
                                  is_training=is_training, padding='SAME', name='conv4')
        # (-1, 60, 16, 128)
        self.max_pool4 = Max_pool(filter_height=1, filter_width=2, stride_y=1, stride_x=2, padding='VALID', name='pool4')
        # (-1, 60, 8, 128)
        self.drop4 = Dropout(self.dropout_keep_prob_cnn)

        # (-1, 60, 8, 128)
        self.conv5 = Conv_bn_relu(filter_height=3, filter_width=3, num_filters=256, stride_y=1, stride_x=1,
                                  is_training=is_training, padding='SAME', name='conv5')
        # (-1, 60, 8, 256)
        self.max_pool5 = Max_pool(filter_height=1, filter_width=2, stride_y=1, stride_x=2, padding='VALID', name='pool5')
        # (-1, 60, 4, 256)
        self.drop5 = Dropout(self.dropout_keep_prob_cnn)

        # (-1, 60, 4, 256)
        self.conv6 = Conv_bn_relu(filter_height=3, filter_width=3, num_filters=256, stride_y=1, stride_x=1,
                                  is_training=is_training, padding='SAME', name='conv6')
        # (-1, 60, 4, 256)
        self.max_pool6 = Max_pool(filter_height=1, filter_width=2, stride_y=1, stride_x=2, padding='VALID',
                                  name='pool6')
        # (-1, 60, 2, 256)
        self.drop6 = Dropout(self.dropout_keep_prob_cnn)

        self.bidirectional = Bidirectional(nhidden=params['rnn_hidden_size'], nlayer=params['rnn_nb_layer'],
                                           input_keep_prob=1., output_keep_prob= self.dropout_keep_prob_rnn,
                                           return_sequences= True, name= 'bidrectional', return_state=False)
        self.attention = SelfAttention(params['attention_size'])

    def call(self, inputs, training):
        self.is_training = training
        x = inputs
        x = self.conv1(x)
        x = self.drop2(self.max_pool2(self.conv2(x)), training = self.is_training)
        x = self.drop3(self.max_pool3(self.conv3(x)), training = self.is_training)
        x = self.drop4(self.max_pool4(self.conv4(x)), training = self.is_training)
        x = self.drop5(self.max_pool5(self.conv5(x)), training = self.is_training)
        x = self.drop6(self.max_pool6(self.conv6(x)), training = self.is_training)
        x = tf.reshape(x, shape=[-1, self.out_shape_sed[0], 2*256])
        x = self.bidirectional(x)
        x = self.attention(x)
        return x

class SED(tf.keras.layers.Layer):
    def __init__(self, params, is_training, out_shape_sed, name = 'SED',
                 dropout_keep_prob_dnn=None, *arg, **kwargs):
        super(SED, self).__init__(name=name, *arg, **kwargs)
        if dropout_keep_prob_dnn is None:
            dropout_keep_prob_dnn = params['dropout_keep_prob_dnn']
        self.dropout_keep_prob_dnn = dropout_keep_prob_dnn
        self.out_shape_sed = out_shape_sed
        self.params = params
        self.is_training = is_training
        self.fc1 = Fc(params['dnn_size'], name='fc1', relu=True)
        self.drop1 = Dropout(self.dropout_keep_prob_dnn, name='drop1')
        self.fc2 = Fc(params['dnn_size'], name='fc2')
        self.drop2 = Dropout(self.dropout_keep_prob_dnn, name='drop2')
        self.score = Fc(out_shape_sed[-1], name='score_sed')
    def call(self, inputs, training):
        self.is_training = training
        x = inputs
        x = self.drop1(self.fc1(x), training = self.is_training)
        x = self.drop2(self.fc2(x), training = self.is_training)
        x = self.score(x)
        x = tf.sigmoid(x)
        x = tf.reshape(x, [-1, self.out_shape_sed[0], self.out_shape_sed[1]])
        return x

class DOA(tf.keras.layers.Layer):
    def __init__(self, params, out_shape_doa, name = 'DOA', is_training = None, *arg, **kwargs):
        super(DOA, self).__init__(name=name, *arg, **kwargs)
        self.out_shape_doa = out_shape_doa
        self.is_training = is_training
        self.params = params
        self.fc1 = Fc(params['dnn_size'], name='fc1')
        self.drop1 = Dropout(params['dropout_keep_prob_dnn'], name='drop1')
        self.fc2 = Fc(params['dnn_size'], name='fc2')
        self.drop2 = Dropout(params['dropout_keep_prob_dnn'], name='drop2')
        self.score = Fc(out_shape_doa[-1], name='score_sed')
    def call(self, inputs, training):
        self.is_training = training
        x = inputs
        x = self.drop1(self.fc1(x), training = self.is_training)
        x = self.drop2(self.fc2(x), training = self.is_training)
        x = self.score(x)
        x = tf.tanh(x)
        x = tf.reshape(x, [-1, self.out_shape_doa[0], self.out_shape_doa[1]])
        return x

class Loss():
    def __init__(self, name=None):
        self.name = name
    def sed_loss_regression(self, pred, gt):
        pred = tf.reshape(pred, shape=[-1, pred.shape[-1]])
        gt = tf.reshape(gt, shape=[-1, gt.shape[-1]])
        # instead of cross-entropy loss, we use mse loss (regression) here
        sed_loss = tf.reduce_mean(tf.square(pred - gt))  # mean in all dimensions
        return sed_loss

    def sed_loss_classification(self, pred, gt):
        pred = tf.reshape(pred, shape=[-1, pred.shape[-1]])
        gt = tf.reshape(gt, shape=[-1, gt.shape[-1]])
        # instead of cross-entropy loss, we use mse loss (regression) here
        sed_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)
        sed_loss = tf.reduce_sum(sed_loss, axis=[1])
        sed_loss = tf.reduce_mean(sed_loss)  # mean in all dimensions
        return sed_loss

    def doa_loss_regression(self, pred_sed, gt_sed, pred_doa, gt_doa):
        pred_sed = tf.reshape(pred_sed, shape=[-1, pred_sed.shape[-1]])
        gt_sed = tf.reshape(gt_sed, shape=[-1, gt_sed.shape[-1]])
        pred_doa = tf.reshape(pred_doa, shape=[-1, pred_doa.shape[-1]])
        gt_doa = tf.reshape(gt_doa, shape=[-1, gt_doa.shape[-1]])

        mask = gt_sed
        mask = tf.concat((mask, mask, mask), axis=-1)

        doa_loss = tf.square(pred_doa - gt_doa)
        doa_loss = tf.multiply(doa_loss, mask)  # mask here
        doa_loss = tf.reduce_sum(doa_loss) / tf.reduce_sum(mask)  # mean in all dimensions
        return doa_loss

    def total_loss(self, pred_sed, gt_sed, pred_doa, gt_doa, sed_loss_weight=1., doa_loss_weight=1.):
        losses =  sed_loss_weight*self.sed_loss_regression(pred= pred_sed, gt= gt_sed) + \
                  doa_loss_weight*self.doa_loss_regression(pred_doa= pred_doa, gt_doa= gt_doa, pred_sed=pred_sed, gt_sed=gt_sed)
        return losses
    def total_loss_l2(self, pred_sed, gt_sed, pred_doa, gt_doa, l2_loss, l2_reg_lambda,
                    sed_weight=1., doa_weight=1.):
        return self.total_loss(pred_sed=pred_sed, gt_sed=gt_sed, pred_doa=pred_doa, gt_doa=gt_doa,
                               sed_loss_weight=sed_weight, doa_loss_weight=doa_weight) + l2_loss*l2_reg_lambda
    def all_loss(self, pred_sed, gt_sed, pred_doa, gt_doa, l2_loss, l2_reg_lambda,
                    sed_weight=1., doa_weight=1.):
        sed_loss = self.sed_loss_regression(pred_sed, gt_sed)
        doa_loss = self.doa_loss_regression(pred_sed, gt_sed, pred_doa, gt_doa)
        total_loss = self.total_loss(pred_sed, gt_sed, pred_doa, gt_doa, sed_weight, doa_weight)
        total_loss_l2 = self.total_loss_l2(pred_sed, gt_sed, pred_doa, gt_doa, l2_loss, l2_reg_lambda, sed_weight, doa_weight)
        return sed_loss, doa_loss, total_loss, total_loss_l2
    def __call__(self):
        return 'Losses'
class SELDnet(tf.keras.Model):
    def __init__(self, params, is_training, out_shape_sed, out_shape_doa, name = None,
                 dropout_keep_prob_cnn=None, dropout_keep_prob_rnn=None, dropout_keep_prob_dnn=None, *args, **kwargs):
        super(SELDnet, self).__init__(name=name, *args, **kwargs)
        if dropout_keep_prob_cnn is None:
            dropout_keep_prob_cnn = params['dropout_keep_prob_cnn']
        self.dropout_keep_prob_cnn = dropout_keep_prob_cnn
        if dropout_keep_prob_rnn is None:
            dropout_keep_prob_rnn = params['dropout_keep_prob_rnn']
        self.dropout_keep_prob_rnn = dropout_keep_prob_rnn
        if dropout_keep_prob_dnn is None:
            dropout_keep_prob_dnn = params['dropout_keep_prob_dnn']
        self.dropout_keep_prob_dnn = dropout_keep_prob_dnn

        self.params = params
        self.is_training = is_training
        self.out_shape_sed = out_shape_sed
        self.out_shape_doa = out_shape_doa
        self.sed_loss = None
        self.doa_loss = None
        self.total_loss = None
        self.crnn = CRNN(is_training= is_training, params= params, name='crnn',
                         out_shape_sed= out_shape_sed, out_shape_doa=out_shape_doa,
                         dropout_keep_prob_cnn=self.dropout_keep_prob_cnn, dropout_keep_prob_rnn=self.dropout_keep_prob_rnn, dropout_keep_prob_dnn=self.dropout_keep_prob_dnn)
        self.sed = SED(params=params, out_shape_sed=out_shape_sed, is_training = is_training)
        self.doa = DOA(params=params, out_shape_doa=out_shape_doa, is_training = is_training)

    def call(self, inputs, training):
        self.is_training = training
        x = inputs
        x = self.crnn(x, training = self.is_training)
        x = tf.reshape(x, [-1, 2*self.params['rnn_hidden_size']])
        x_sed = self.sed(x, training= self.is_training)
        x_doa = self.doa(x, training= self.is_training)
        return x_sed, x_doa

from parameter import *
path = '/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev/fold1_room1_mix007_ov1.npy'
path_label = '/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev_label/fold1_room1_mix007_ov1.npy'

x = np.load(path)
x = tf.reshape(x, (-1, 300, 64, 7))
params = get_params()
#layer = CRNN(is_training=True, params=params, name= 'crnn', out_shape_sed=(60, 14), out_shape_doa=(60, 42))
layer = SELDnet(params=params, is_training=True, out_shape_sed=(60,14), out_shape_doa=(60,42))

y = np.load(path_label)
y_sed = tf.constant(y[:, :14], dtype=tf.float32)
y_doa = tf.constant(y[:, 14:], dtype=tf.float32)
print(y_sed.shape,' ', y_doa.shape)

pred_sed, pred_doa = layer(x, training = False)
l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in layer.trainable_variables])
losses = Loss()

with tf.GradientTape(persistent=True) as tape:
    pred_sed, pred_doa = layer(x, training=True)
    loss = losses.total_loss_l2(pred_sed=pred_sed, gt_sed=y_sed, pred_doa=pred_doa,
                                gt_doa=y_doa, l2_loss=l2_loss, l2_reg_lambda=params['l2_reg_lambda'])
grads = tape.gradient(loss, layer.trainable_variables)
for var in tape.watched_variables():
    print(var.name)
print(loss)

