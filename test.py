import shutil
import tensorflow as tf
import os
import sys
import numpy as np
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from parameter import *
from cls_feature_class import *
from dataregulator import *
from play_2 import *
from augmentation import *
from metrics.evaluation_metrics import *
from metrics.SELD_evaluation_metrics import *

tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.compat.v1.flags.DEFINE_integer("task_id", 4, "evaluation or development?")
tf.compat.v1.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
tf.compat.v1.flags.DEFINE_integer("augment", 0, "Augmentation")

tf.compat.v1.flags.DEFINE_float("learning_rate", 0.0002, "Numer of training step to evaluate (default: 100)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.5, "Numer of training step to evaluate (default: 100)")
tf.compat.v1.flags.DEFINE_integer("training_epoch", 2000, "Numer of training step to evaluate (default: 100)")

tf.compat.v1.flags.DEFINE_integer("evaluate_every", 100, "Numer of training step to evaluate (default: 100)")

tf.compat.v1.flags.DEFINE_integer("seq_len", 600, "Feature sequence length (default: 300)")

tf.compat.v1.flags.DEFINE_integer("early_stopping", 0, "Early stopping (default: 0)")
tf.compat.v1.flags.DEFINE_integer("patience", 50,
                                  "Number of evaluation without improvement to trigger early stopping (default: 50)")

FLAGS = tf.compat.v1.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):  # python3
    print("{} = {}".format(attr.upper(), value))
print("")

print("augment")
print(FLAGS.augment)
print("learning_rate")
print(FLAGS.learning_rate)
print("decay rate")
print(FLAGS.decay_rate)
print("training_epoch")
print(FLAGS.training_epoch)

# path where some output are stored
# os.path.abspath means that this path will be where the file run(but WE CAN CHANGE THE DIR BY USING os.chdir(dir))
out_path = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path, FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

evaluate_every = FLAGS.evaluate_every
seq_len = FLAGS.seq_len

# learning schedule
scheduler = dict(
    learning_rate=FLAGS.learning_rate,
    decay_rate=FLAGS.decay_rate,
    warmup_epoch=10,
    schedule=[200, 600, 1000],
    training_epoch=FLAGS.training_epoch
)

'''audio_path = os.path.abspath('fold1_room1_mix001_ov1.wav')
audio, fs = feat_class._load_audio(audio_path)
print(audio.shape)
spectrogram = feat_class._spectrogram(audio)
foa_iv = feat_class._get_foa_intensity_vectors(spectrogram)
gcc_feat = feat_class._get_gcc(spectrogram)
spectrogram = feat_class._get_mel_spectrogram(spectrogram)
print(spectrogram.shape)
print(foa_iv.shape)
print(gcc_feat.shape)
print(feat_class._max_label_frames)'''
# plt.plot(spectrogram[:,:])
# plt.show()

params = get_params(str(FLAGS.task_id))
feat_class = FeatureClass(params)
'''feat_class.extract_all_feature()
feat_class.preprocess_features()
feat_class.extract_all_labels()'''

train_splits, val_splits, test_splits = None, None, None

if params['mode'] == 'dev':
    test_splits = [1]
    val_splits = [2]
    train_splits = [3, 4, 5, 6]

elif params['mode'] == 'eval':
    test_splits = [7, 8]
    val_splits = [1]
    train_splits = [2, 3, 4, 5, 6]

iseval = (params['mode'] == 'eval')

data_gen_train = DataRegulator(train_splits,
                               params['feat_label_dir'] + '{}_dev_label/'.format(params['dataset']),
                               params['feat_label_dir'] + '{}_dev_norm/'.format(params['dataset']),
                               seq_len=seq_len,
                               seq_hop=5)  # hop len must be factor of 5
data_gen_valid = DataRegulator(val_splits,
                               params['feat_label_dir'] + '{}_dev_label/'.format(params['dataset']),
                               params['feat_label_dir'] + '{}_dev_norm/'.format(params['dataset']),
                               seq_len=seq_len,
                               seq_hop=seq_len)

data_gen_test = DataRegulator(test_splits,
                              params['feat_label_dir'] + '{}_{}_label/'.format(params['dataset'], params['mode']),
                              params['feat_label_dir'] + '{}_{}_norm/'.format(params['dataset'], params['mode']),
                              seq_len=seq_len,
                              seq_hop=seq_len,
                              iseval=iseval)

data_gen_test.load_data()
data_gen_valid.load_data()
data_gen_train.load_data()
data_gen_train.shuffle_data()

best_valid_seld_metric = np.inf
early_stop_count = 0
num_batch_each_file = 3000//seq_len
out_shape_sed = data_gen_train.get_out_shape_sed()
out_shape_doa = data_gen_train.get_out_shape_doa()

def learning_schedule(epoch):
    divide_epoch = np.array(scheduler['schedule'])
    decay = sum(epoch >= divide_epoch)
    if epoch <= scheduler['warmup_epoch']:
        return scheduler['learning_rate'] * 0.1
    return scheduler['learning_rate'] * np.power(scheduler['decay_rate'], decay)

def eval_step(params, x_mel, y_sed, y_doa):
    if FLAGS.augment == 1:
        x_mel = tf.constant(augment_spec(x_mel), dtype=tf.float32)
    else:
        x_mel = tf.dtypes.cast(x_mel, tf.float32)

    y_sed = tf.dtypes.cast(y_sed, tf.float32)
    y_doa = tf.dtypes.cast(y_doa, tf.float32)

    pred_sed, pred_doa = model(x_mel,training = False)

    loss = Loss()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
    loss_sed,loss_doa,loss_total,loss_total_l2 = loss.all_loss(pred_sed, y_sed, pred_doa, y_doa, l2_loss,
                                                               params['l2_reg_lambda'], params['loss_weights'][0], params['loss_weights'][1])

    dic = dict(loss_sed=loss_sed, loss_doa=loss_doa, loss_total=loss_total,
               loss_total_l2=loss_total_l2, score_sed=pred_sed, score_doa=pred_doa)
    return dic

def evaluate(dataset, num_files, params=None):
    # Validate the model on the entire evaluation test set after each epoch
    loss_total = 0
    loss_sed = 0
    loss_doa = 0
    loss_total_l2 = 0

    N = num_batch_each_file #5
    score_sed = np.zeros(np.append([num_files * N], out_shape_sed))
    score_doa = np.zeros(np.append([num_files * N], out_shape_doa))

    test_batch_size = params['batch_size'] // 8 #8

    for test_step, (x_mel, y_sed, y_doa) in enumerate(dataset):
        ret = eval_step(params, x_mel, y_sed, y_doa)
        if (test_step + 1) * test_batch_size <= num_files * N:
            score_sed[test_step * test_batch_size : (test_step + 1) * test_batch_size] = ret['score_sed']
            score_doa[test_step * test_batch_size : (test_step + 1) * test_batch_size] = ret['score_doa']
        else:
            score_sed[test_step * test_batch_size : num_files * N] = ret['score_sed']
            score_doa[test_step * test_batch_size : num_files * N] = ret['score_doa']
        loss_sed += ret['loss_sed']
        loss_doa += ret['loss_doa']
        loss_total += ret['loss_total']
        loss_total_l2 += ret['loss_total_l2']

    return loss_sed, loss_doa, loss_total, loss_total_l2, score_sed, score_doa

def metric_dcase2020(gen, sed_pred, doa_pred):
    sed_gt = gen.all_label_sed_2d()
    doa_gt = gen.all_label_doa_2d()
    cls_new_metric = SELDMetrics(nb_classes=gen._Ncat, doa_threshold=params['lad_doa_thresh'])
    pred_dict = feat_cls.regression_label_format_to_output_format(sed_pred, doa_pred)
    gt_dict = feat_cls.regression_label_format_to_output_format(sed_gt, doa_gt)
    pred_blocks_dict = feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
    gt_blocks_dict = feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

    cls_new_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
    new_metric = cls_new_metric.compute_seld_scores()
    new_seld_metric = early_stopping_metric(new_metric[:2], new_metric[2:])

    return new_metric, new_seld_metric

def log_file(filename, loss_sed, loss_doa, loss_total, new_metric, new_seld_metric):
    with open(os.path.join(out_path, filename), "a") as text_file:
        text_file.write("{:0.5f} {:0.5f} {:0.5f} ".format(loss_sed, loss_doa, loss_total))
        # dcase 2020
        text_file.write("{:0.4f} {:0.2f} ".format(new_metric[2], new_metric[3] * 100))
        text_file.write("{:0.4f} {:0.2f} ".format(new_metric[0], new_metric[1] * 100))
        text_file.write("{:0.4f}\n".format(new_seld_metric))

def print_metric(new_metric, new_seld_metric):
    s = '2020 DOA-ER: {:0.4f} '.format(new_metric[2])
    s += 'Fr-Recall: {:0.2f} '.format(new_metric[3] * 100)
    s += 'SED-ER: {:0.4f} '.format(new_metric[0])
    s += 'SED-F1: {:0.2f} '.format(new_metric[1] * 100)
    s += 'SELD: {:0.4f} '.format(new_seld_metric)
    print(s)

def train_step(params, step, x_mel, y_sed, y_doa, LR):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    if FLAGS.augment == 1:
        x_mel = tf.constant(augment_spec(x_mel), dtype=tf.float32)
    else:
        x_mel = tf.dtypes.cast(x_mel, tf.float32)

    y_sed = tf.dtypes.cast(y_sed, tf.float32)
    y_doa = tf.dtypes.cast(y_doa, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        pred_sed, pred_doa = model(x_mel, training=True)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss = Loss()
        train_loss_sed, train_loss_doa, train_loss_total, train_loss_total_l2 = loss.all_loss(pred_sed, y_sed, pred_doa, y_doa, l2_loss,
                                                                                              params['l2_reg_lambda'], params['loss_weights'][0], params['loss_weights'][1])
    grads = tape.gradient(train_loss_total_l2, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    time_str = datetime.now().isoformat()
    print("{}: At step {} ~ loss SED {} DOA {} OUTPUT {} TOTAL {}".format(time_str, step, train_loss_sed, train_loss_doa,
                                                                      train_loss_total, train_loss_total_l2))
    return train_loss_sed, train_loss_doa, train_loss_total, train_loss_total_l2

def train_loop(params, scheduler, dataset_train):
    start_time = time.time()
    #train_batches_per_epoch = np.floor(data_gen_train._data_size / params['batch_size']).astypnp.uint32
    current_step = 0
    global early_stop_count, best_valid_seld_metric
    for epoch in range(scheduler['training_epoch']):
        if epoch==20:
            break
        applied_LR = learning_schedule(epoch)
        print('\n----------------------------------------------------------')
        print("Epoch {} ~ learning rate {}".format(epoch + 1, applied_LR)) #datetime.now(),

        for step, (x_mel, y_sed, y_doa) in enumerate(dataset_train):
            train_loss_sed, train_loss_doa,\
            train_loss_total, train_loss_total_l2 = train_step(step= step, params=params, LR= applied_LR,
                                                               x_mel=x_mel, y_sed=y_sed, y_doa=y_doa)
            current_step +=1
            if current_step % evaluate_every:
                print("%%%%%% {} Start validation".format(datetime.now()))
                valid_loss_sed, valid_loss_doa, valid_loss_total, \
                valid_loss_total_l2, valid_score_sed, valid_score_doa = \
                    evaluate(dataset_valid, data_gen_valid._data_size, params=params)
                test_loss_sed, test_loss_doa, test_loss_total, \
                test_loss_l2, test_score_sed, test_score_doa = \
                    evaluate(dataset_test, data_gen_test._data_size, params=params)

                valid_score_sed = reshape_3Dto2D(valid_score_sed) > 0.5
                valid_doa_pred = reshape_3Dto2D(valid_score_doa)
                test_score_sed = reshape_3Dto2D(test_score_sed) > 0.5
                test_doa_pred = reshape_3Dto2D(test_score_doa)

                valid_new_metric, valid_new_seld_metric = \
                    metric_dcase2020(data_gen_valid, valid_score_sed, valid_doa_pred)
                test_new_metric, test_new_seld_metric = \
                    metric_dcase2020(data_gen_test, test_score_sed, test_doa_pred)

                log_file("valid_metric.txt", valid_loss_sed, valid_loss_doa, valid_loss_total,
                         valid_new_metric, valid_new_seld_metric)
                log_file("test_metric.txt", test_loss_sed, test_loss_doa, test_loss_total,
                         test_new_metric, test_new_seld_metric)

                print_metric(valid_new_metric, valid_new_seld_metric)
                print_metric(test_new_metric, test_new_seld_metric)

                early_stop_count += 1
                if valid_new_seld_metric < best_valid_seld_metric:
                    early_stop_count = 0  # reset
                    best_valid_seld_metric = valid_new_seld_metric
                    # save the last model
                    manager.save(checkpoint_number= current_step)

                    print("Best model seld metric updated")
                    source_file = os.path.join(checkpoint_path,'ckpt-'+str(current_step))
                    dest_file = os.path.join(checkpoint_path, 'best_model_seld')
                    shutil.copy(source_file + '.data-00000-of-00001', dest_file + '.data-00000-of-00001')
                    shutil.copy(source_file + '.index', dest_file + '.index')
                    shutil.copy(checkpoint_path+'/checkpoint', checkpoint_path + '/checkpoint_'+str(current_step))

                    log_file("current_best_valid.txt", valid_loss_sed, valid_loss_doa, valid_loss_total,
                             valid_new_metric, valid_new_seld_metric)
                    log_file("current_best_test.txt", test_loss_sed, test_loss_doa, test_loss_total,
                             test_new_metric, test_new_seld_metric)

                data_gen_valid.reset_pointer()
                data_gen_test.reset_pointer()

                if (FLAGS.early_stopping == 1 and early_stop_count >= FLAGS.patience):
                    end_time = time.time()
                    with open(os.path.join(out_path, "training_time.txt"), "a") as text_file:
                        text_file.write("{:g}\n".format((end_time - start_time)))
                    quit()

        data_gen_train.reset_pointer()
        data_gen_train.shuffle_data()

    end_time = time.time()
    with open(os.path.join(out_path, "training_time.txt"), "a") as text_file:
        text_file.write("{:g}\n".format((end_time - start_time)))


#model = SELDnet(params=params, is_training=True, out_shape_sed=(120,14), out_shape_doa=(120,42))

'''x_mel, y_sed, y_doa = data_gen_train.next_batch(params['batch_size'])
y_sed = tf.constant(y_sed, dtype=tf.float32)
y_doa = tf.constant(y_doa, dtype=tf.float32)
print(x_mel.shape)
print(y_doa.shape)
print(y_sed.shape)

feat_path='/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev_norm/fold1_room1_mix007_ov1.npy'
label_path='/home/ad/PycharmProjects/Sound_processing/venv/pull_data/feat_label/foa_dev_label/fold1_room1_mix007_ov1.npy'

x_mel = np.load(feat_path).reshape((-1, 600, 64, 7))

x_mel = tf.constant(augment_spec(x_mel), dtype=tf.float32)
y = np.load(label_path)
y_sed = tf.constant(y[:,:14], dtype=tf.float32)
y_doa = tf.constant(y[:,14:], dtype=tf.float32)

print(x_mel.shape)
print(y_sed.shape)
print(y_doa.shape)
'''
model = SeldNet.build(params=params, out_shape_sed=(120,14), out_shape_doa=(120,42), seq_length=600)
model.summary()
ckpt = tf.train.Checkpoint(step = tf.Variable(0), net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

#get dataset_train:
data_gen_train.reset_pointer()
actual_len_train, inputs, label_sed, label_doa = data_gen_train.get_rest_batch()
dataset_train = tf.data.Dataset.from_tensor_slices((inputs, label_sed, label_doa)).batch(params['batch_size'])

data_gen_valid.reset_pointer()
actual_len_valid, inputs, label_sed, label_doa = data_gen_valid.rest_batch_whole()
dataset_valid = tf.data.Dataset.from_tensor_slices((inputs, label_sed, label_doa)).batch(params['batch_size']//8)

data_gen_test.reset_pointer()
actual_len_test, inputs, label_sed, label_doa = data_gen_test.rest_batch_whole()
dataset_test = tf.data.Dataset.from_tensor_slices((inputs, label_sed, label_doa)).batch(params['batch_size']//8)

train_loop(params, scheduler, dataset_train)
