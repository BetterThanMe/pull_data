import tensorflow as tf
import os
import sys
sys.path.insert(0, '/home/ad/PycharmProjects/Sound_processing/venv/pull_data/')
from parameter import *

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