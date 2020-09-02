import pandas as pd
import os
import tensorflow as tf
import numpy as np
import shutil

import librosa
import tensorflow as tf
import os
import sys
import numpy as np
from datetime import datetime
from time import time
from parameter import *
from cls_feature_class import *
from dataregulator import *
from play_2 import *
from augmentation import *

params = get_params(str(4))
feat_class = FeatureClass(params)
splits = [1,2,3,4,5,6]
gen = DataRegulator(splits,
                    params['feat_label_dir'] + '{}_dev_label/'.format(params['dataset']),
                    params['feat_label_dir'] + '{}_dev_norm/'.format(params['dataset']),
                    seq_len=600,
                    seq_hop=5)
gen.load_data()
gen.shuffle_data()

model = SELDnet(params=params, is_training=True, out_shape_sed=(120,14), out_shape_doa=(120,42))

def train_step(x_mel, y_sed, y_doa):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    loss = Loss()

    with tf.GradientTape() as tape:

        pred_sed, pred_doa = model(x_mel, training=True)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

        #train_loss_sed = loss.sed_loss_regression(pred_sed, y_sed)
        #train_loss_doa = loss.doa_loss_regression(pred_sed, y_sed, pred_doa, y_doa)
        train_loss_total = loss.total_loss(pred_sed, y_sed, pred_doa, y_doa, 1., 1.)
        #train_loss_total_l2 = loss.total_loss_l2(pred_sed, y_sed, pred_doa, y_doa, l2_loss, params['l2_reg_lambda'], 1., 1.)

    grads = tape.gradient(train_loss_total, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    time_str = datetime.now().isoformat()

    print('train_loss_sed: ', train_loss_total)

    #print("{}:loss SED {} DOA {} OUTPUT {} TOTAL {}".format(time_str, train_loss_sed, train_loss_doa,
    #                                                                  train_loss_total, train_loss_total_l2))
    #return train_loss_sed, train_loss_doa, train_loss_total, train_loss_total_l2

def train_loop(num_epochs):
    gen.reset_pointer()
    for epoch in range(num_epochs):
        x_mel, y_sed, y_doa = gen.next_batch(params['batch_size'])
        print('=======START======')
        print('pointer: ', gen._pointer)
        print(np.unique(y_sed))

        x_mel = tf.constant(augment_spec(x_mel), dtype=np.float32)
        y_sed = tf.constant(y_sed, dtype=np.float32)
        y_doa = tf.constant(y_doa, dtype=np.float32)
    
        print('++++++++++++++++')
        print(max(np.unique(y_sed.numpy())))
        print(min(np.unique(y_sed.numpy())))
        print('---------------')
        train_step(x_mel=tf.identity(x_mel), y_sed=tf.identity(y_sed), y_doa=tf.identity(y_doa))
        print('=====FINISH=====================')


optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = Loss()

data = []
for i in range(20):
    dic = dict()
    x_mel, y_sed, y_doa = gen.next_batch(params['batch_size'])
    dic['mel'] = tf.constant(augment_spec(x_mel), dtype=np.float32)
    dic['sed'] = tf.constant(y_sed, dtype=np.float32)
    dic['doa'] = tf.constant(y_doa, dtype=np.float32)
    #print(np.unique(dic['mel'].numpy()))
    #print(np.unique(dic['sed'].numpy()))
    #print(np.unique(dic['doa'].numpy()))
    data.append(dic)
print('==========================')

for i in data:
    #print(np.unique(i['mel'].numpy()))
    #print(np.unique(i['sed'].numpy()))
    #print(np.unique(i['doa'].numpy()))

    with tf.GradientTape() as tape:
        pred_sed, pred_doa = model(i['mel'], training = True)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        train_loss_total = loss.total_loss(pred_sed, i['sed'], pred_doa, i['doa'], 1., 1.) + l2_loss*params['l2_reg_lambda']
    grads = tape.gradient(train_loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print('total loss = ', train_loss_total)
    print('--------------------------------------------------')


'''mask = tf.zeros([600,14])
mask = tf.concat((mask, mask, mask), axis=-1)

print(tf.reduce_sum(mask))
'''
'''
gen.reset_pointer()
for i in range(5):
    x_mel, y_sed, y_doa = gen.next_batch(params['batch_size'])
    x_mel_t = tf.constant(augment_spec(x_mel), dtype=np.float32)
    y_sed_t = tf.constant(y_sed, dtype=np.float32)
    y_doa_t = tf.constant(y_doa, dtype=np.float32)
    print(max(np.unique(y_sed_t.numpy())))
    print(min(np.unique(y_sed_t.numpy())))

    with tf.GradientTape() as tape:

        pred_sed, pred_doa = model(x_mel_t, training=True)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

        #train_loss_sed = loss.sed_loss_regression(pred_sed, y_sed)
        #train_loss_doa = loss.doa_loss_regression(pred_sed, y_sed, pred_doa, y_doa)
        train_loss_total = loss.total_loss(pred_sed, y_sed_t, pred_doa, y_doa_t, 1., 1.)
        #train_loss_total_l2 = loss.total_loss_l2(pred_sed, y_sed, pred_doa, y_doa, l2_loss, params['l2_reg_lambda'], 1., 1.)

    grads = tape.gradient(train_loss_total, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #time_str = datetime.now().isoformat()

    print('train_loss_sed: ', train_loss_total)
'''