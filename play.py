import librosa
import os
import librosa.display
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
audio_path = os.path.abspath('split1_ir3_ov2_73.wav')
audio, fs = librosa.load(audio_path)
audio_trimed, _ = librosa.effects.trim(audio)
n_fft = 2048

D = np.abs(librosa.stft(audio_trimed, n_fft= n_fft, hop_length= int(n_fft/2)))
D = librosa.amplitude_to_db(D, ref= np.max)
librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear')

n_mels = 128
S = librosa.feature.melspectrogram(audio_trimed, sr=fs, n_fft=n_fft, hop_length=int(n_fft/2), n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=fs, hop_length=int(n_fft/2), x_axis='time', y_axis='mel');

plt.colorbar()
#plt.show()"""

path = os.path.abspath('split1_ir0_ov1_1.csv')
#_desc_file =  pd.read_csv(os.path.abspath('split1_ir0_ov1_1.csv'))

import traceback
import contextlib
@contextlib.contextmanager
def assert_raise(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception {}'.format(error_class))
        traceback.print_exc(limit= 2)
    except Exception as e:
        raise e
    else:
        raise Exception('Expected {} to be raised but no error was raised'.format(error_class))

@tf.function
def pow(a,b):
    return a**b

square = pow.get_concrete_function(a = tf.TensorSpec(None, dtype= tf.float32), b = tf.TensorSpec(None, dtype= tf.float32))
print(square)

#assert square(tf.constant(10.)) == 100

graph = square.graph
for node in graph.as_graph_def().node:
    print(f'{node.input} -> {node.name}')
print(tf.GraphKeys.GLOBAL_VARIABLES)

'''@tf.function
def fizzbuzz(n):
  for i in tf.range(1, n + 1):
    print('Tracing for loop')
    if i % 15 == 0:
      print('Tracing fizzbuzz branch')
      tf.print('fizzbuzz')
    elif i % 3 == 0:
      print('Tracing fizz branch')
      tf.print('fizz')
    elif i % 5 == 0:
      print('Tracing buzz branch')
      tf.print('buzz')
    else:
      print('Tracing default branch')
      tf.print(i)

fizzbuzz(tf.constant(5))
fizzbuzz(tf.constant(20))'''