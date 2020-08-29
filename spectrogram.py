from librosa import display
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

audio_path = os.path.abspath('split1_ir3_ov2_73.wav')
audio, sr = librosa.load(audio_path)

print(audio.strides)
print(audio.shape)
print(sr)
def spectrogram(signal, sample_rate, hop_len_s = 0.01, window_s = 0.02, max_freq = None, eps = 1e-14):
    stride_size = int(sample_rate* hop_len_s)
    window_size = int(sample_rate* window_s)

    truncate_size = (len(signal) - window_size) % stride_size +1
    #signal = np.concatenate((signal, np.zeros(len(signal)-truncate_size)))
    signal = signal[:len(signal) - truncate_size]
    #nshape[window_size, num_windows]
    nshape = (window_size, (len(signal) - window_size)// stride_size + 1)
    nstrides = (signal.strides[0], signal.strides[0]*stride_size)
    windows = np.lib.stride_tricks.as_strided(signal,
                                              shape= nshape, strides= nstrides)
    #assert np.all(windows[:, 1]) == signal[stride_size: (stride_size+window_size)]
    print('stride_side = ', stride_size)
    print('nshape', nshape)

    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows*weighting, axis= 0)
    fft = np.absolute(fft)
    fft = fft**2

    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *=(2./scale)
    fft[(0,-1), :]/=scale

    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    ind = np.where(freqs<=max_freq)[0][-1] +1
    spectro = np.log(fft[:ind, :]+ eps)
    return spectro
spectrogram_eg = spectrogram(audio, sample_rate=sr, max_freq= 11025)
print(spectrogram_eg.shape)
print(type(spectrogram_eg))
plt.plot(spectrogram_eg.T)
plt.show()