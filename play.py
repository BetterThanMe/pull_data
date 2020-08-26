import librosa
import os

audio_path = os.path.abspath('split1_ir3_ov2_73.wav')
fs, audio = librosa.load(audio_path)
sthing, _ = librosa.effects.trim(fs)
librosa.display.waveplot(sthing, sr = audio)



