import librosa
import numpy as np
import matplotlib.pyplot as plt

data_path = '../Data/VOICED/voice/voice001.txt'
root = '../Data'
sr = 8000

if __name__ == '__main__':
    data = np.genfromtxt(data_path)[1000:11000]
    data_shift = librosa.effects.pitch_shift(data, sr=sr, n_steps=6.0)
    np.savetxt(root+'pitch_shift.txt', data_shift)
