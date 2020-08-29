import librosa
import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/ad/PycharmProjects/Sound_processing/venv/pull_data/')
from parameter import *
from cls_feature_class import *
from dataregulator import *

tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.compat.v1.flags.DEFINE_integer("task_id", 999, "evaluation or development?")
tf.compat.v1.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")
tf.compat.v1.flags.DEFINE_integer("augment", 0, "Augmentation")

tf.compat.v1.flags.DEFINE_float("learning_rate", 0.0002, "Numer of training step to evaluate (default: 100)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.5, "Numer of training step to evaluate (default: 100)")
tf.compat.v1.flags.DEFINE_integer("training_epoch", 2000, "Numer of training step to evaluate (default: 100)")

tf.compat.v1.flags.DEFINE_integer("evaluate_every", 100, "Numer of training step to evaluate (default: 100)")

tf.compat.v1.flags.DEFINE_integer("seq_len", 600, "Feature sequence length (default: 300)")

tf.compat.v1.flags.DEFINE_integer("early_stopping", 0, "Early stopping (default: 0)")
tf.compat.v1.flags.DEFINE_integer("patience", 50, "Number of evaluation without improvement to trigger early stopping (default: 50)")

FLAGS = tf.compat.v1.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()): # python3
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
#os.path.abspath means that this path will be where the file run(but WE CAN CHANGE THE DIR BY USING os.chdir(dir))
out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
# path where checkpoint models are stored
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))
if not os.path.isdir(os.path.abspath(out_path)): os.makedirs(os.path.abspath(out_path))
if not os.path.isdir(os.path.abspath(checkpoint_path)): os.makedirs(os.path.abspath(checkpoint_path))

evaluate_every = FLAGS.evaluate_every
seq_len = FLAGS.seq_len

#learning schedule
scheduler = dict(
learning_rate = FLAGS.learning_rate,
decay_rate = FLAGS.decay_rate,
warmup_epoch = 10,
schedule = [200, 600, 1000],
training_epoch = FLAGS.training_epoch
)

params = get_params(str(FLAGS.task_id))
feat_class = FeatureClass(params)
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
#plt.plot(spectrogram[:,:])
#plt.show()

feat_class.extract_all_feature()
feat_class.preprocess_features()
feat_class.extract_all_labels()
'''params = get_params(str(FLAGS.task_id))
feat_cls = FeatureClass(params)

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
                               seq_hop=5) # hop len must be factor of 5
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
#data_gen_train.load_data()
#data_gen_valid.load_data()
data_gen_test.load_data()
#data_gen_train.shuffle_data()
print(data_gen_test._data_size)
print(data_gen_test._Ncat)

best_valid_seld_metric = np.inf
early_stop_count = 0'''