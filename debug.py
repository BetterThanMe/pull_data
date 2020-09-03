import shutil
import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from time import time
from parameter import *
from dataregulator import *
from play_2 import *
from augmentation import *

def sed_loss_regression(pred, gt):
    pred = tf.reshape(pred, shape=[-1, pred.shape[-1]])
    gt = tf.reshape(gt, shape=[-1, gt.shape[-1]])
    # instead of cross-entropy loss, we use mse loss (regression) here
    sed_loss = tf.reduce_mean(tf.square(pred - gt))  # mean in all dimensions
    return sed_loss

def doa_loss_regression(pred_sed, gt_sed, pred_doa, gt_doa):
    pred_sed = tf.reshape(pred_sed, shape=[-1, pred_sed.shape[-1]])
    gt_sed = tf.reshape(gt_sed, shape=[-1, gt_sed.shape[-1]])
    pred_doa = tf.reshape(pred_doa, shape=[-1, pred_doa.shape[-1]])
    gt_doa = tf.reshape(gt_doa, shape=[-1, gt_doa.shape[-1]])

    mask = gt_sed
    mask = tf.concat((mask, mask, mask), axis=-1)

    doa_loss = tf.square(pred_doa - gt_doa)
    doa_loss = tf.multiply(doa_loss, mask)  # mask here
    if tf.reduce_sum(mask) !=0.:
        doa_loss = tf.reduce_sum(doa_loss) / tf.reduce_sum(mask)  # mean in all dimensions
    else:
        doa_loss = tf.reduce_sum(doa_loss)
    return doa_loss

def total_loss(pred_sed, gt_sed, pred_doa, gt_doa, sed_loss_weight=1., doa_loss_weight=1.):
    losses =  sed_loss_weight*sed_loss_regression(pred= pred_sed, gt= gt_sed) + \
              doa_loss_weight*doa_loss_regression(pred_doa= pred_doa, gt_doa= gt_doa, pred_sed=pred_sed, gt_sed=gt_sed)
    return losses

#--------------------------------------------------------------------------------------------------------------------

params = get_params(str(4))
splits = [3,4,5,6]
gen = DataRegulator(splits,
                    params['feat_label_dir'] + '{}_dev_label/'.format(params['dataset']),
                    params['feat_label_dir'] + '{}_dev_norm/'.format(params['dataset']),
                    seq_len=600,
                    seq_hop=5)
gen.load_data()
#gen.shuffle_data()

#model = SELDnet(params=params, is_training=False, out_shape_sed=(120,14), out_shape_doa=(120,42))

model = SeldNet.build(params=params, out_shape_sed=(120,14), out_shape_doa=(120,42), seq_length=600)

actual_len, x_mel, y_sed, y_doa = gen.get_rest_batch()

#model.compile(optimizer='adam', loss= "mse")

#model.fit()

'''optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

gen.reset_pointer()
for i in range(5):
    x_mel, y_sed, y_doa = gen.next_batch(params['batch_size'])
    x_mel_t = tf.constant(augment_spec(x_mel), dtype=np.float32)
    y_sed_t = tf.constant(y_sed, dtype=np.float32)
    y_doa_t = tf.constant(y_doa, dtype=np.float32)

    print('x_mel: ', max(np.unique(x_mel)), ' - ', min(np.unique(x_mel)))
    print('y_sed: ', max(np.unique(y_sed)), ' - ', min(np.unique(y_sed)))
    print('y_doa: ', max(np.unique(y_doa)), ' - ', min(np.unique(y_doa)))

    with tf.GradientTape() as tape:

        pred_sed, pred_doa = model(x_mel_t, training=True)

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])

        train_loss_sed = sed_loss_regression(pred_sed, y_sed_t)
        train_loss_doa = doa_loss_regression(pred_sed, y_sed_t, pred_doa, y_doa_t)
        train_loss_total = total_loss(pred_sed, y_sed_t, pred_doa, y_doa_t, 1., 1.)
        #train_loss_total_l2 = loss.total_loss_l2(pred_sed, y_sed, pred_doa, y_doa, l2_loss, params['l2_reg_lambda'], 1., 1.)

    grads = tape.gradient(train_loss_total, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #time_str = datetime.now().isoformat()

    print('loss: SED {} | DOA {} | TOTAL {} '.format(train_loss_sed, train_loss_doa, train_loss_total))
    print()
'''
#-----------------------------------------------------------------------
'''
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
'''