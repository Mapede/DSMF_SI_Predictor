# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:09:47 2021

@author: mbp
"""
import numpy as np
from scipy.io.wavfile import read
from functions import pre_process, load_Network_weights

# input your .wav file paths here
s_path = 'your_clean_speech_file.wav'
x_path = 'your_processed_speech_file.wav'

weights_path = 'Network_weights.h5'

_, s = read(s_path)
fs, x = read(x_path)

S, X = pre_process(s, x, fs)

model = load_Network_weights(weights_path, dsmf=False)

# compute and print the raw (unmapped) network output.
i = np.squeeze(model.predict([S, X]))
print('Network output: {}'.format(i))