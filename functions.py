# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:03:34 2021

@author: mbp
"""
import h5py
import numpy as np
from scipy.signal import hann, resample_poly
import network as net

def thirdoct(fs, fft_size, numBands, mn):
    '''
    compute the 1/3 octave band matrix
    
    fs: sampling frequency
    fft_size: size of the fft window
    numBands: number of desired 1/3 octave bands
    mn: the first band center frequency
    
    A: 1/3 octave band matrix
    cf: center frequency vector
    flr: the left and right edge bands
    '''
    f = np.linspace(0, fs, fft_size+1)
    f = f[:int(fft_size/2+1)]
    k = np.arange(numBands)
    cf = 2**(k/3)*mn
    
    fl = np.sqrt((2**(k/3)*mn) * 2**((k-1)/3)*mn)
    fr = np.sqrt((2**(k/3)*mn) * 2**((k+1)/3)*mn)
    flr = []
    A = np.zeros((numBands, len(f)))
    
    for i in range(len(cf)):
        b = np.argmin((f-fl[i])**2)
        fl[i] = f[b]
        fl_ii = b
        
        b = np.argmin((f-fr[i])**2)
        fr[i] = f[b]
        fr_ii = b
        A[i, np.arange(fl_ii, fr_ii)] = 1
        
        flr.append((fl_ii,fr_ii))
    
    rnk = np.sum(A, axis = 1)
    numBands = np.nonzero(np.logical_and((rnk[1:] >= rnk[:-1]), (rnk[1:] != 0)) != 0)[-1][-1]+2
    A = A[:numBands, :]
    cf = cf[:numBands]
    return A, cf, flr

def transform_matrix(x, A, frame_size, overlap, fs, fft_size=512):
    '''
    compute the 1/3 octave band representation of x
    
    x: input signal
    A: 1/3 octave band matrix
    frame_size: size of the sliding window
    overlap: number of samples to shift the window
    fs: sampling frequency of x
    fft_size: size of the fft window
    
    X: The 1/3 octave band representation of x
    '''
    if fs != 20000:
        x = resample_poly(x, 20000, fs)
    
    frames = np.arange(0, len(x)-frame_size+1, overlap)
    x_hat = np.zeros((len(frames), fft_size), dtype=np.complex128)
    
    w = hann(frame_size+1, False)[1:]
    
    for i in range(len(frames)):
        ii = np.arange(frames[i], frames[i] + frame_size)
        x_hat[i, :] = np.fft.fft(x[ii]*w, n=fft_size)
        
    numBands = np.size(A, axis=0)
    x_hat = x_hat[:,:int(fft_size/2 + 1)].T
    
    X = np.zeros((numBands, np.size(x_hat, axis=1)))
    
    for i in range(np.size(x_hat, axis=1)):
        X[:,i] = np.sqrt(np.dot(A, np.abs(x_hat[:,i])**2))
    
    return X

def pre_process(s, x, fs):
    '''
    Preprocessing of time-domain waveform signals s and x, clean and noisy repectively.
    
    s: clean reference speech signal.
    x: noisy/processed speech signal
    fs: sampling frequency of s and x
    
    S: 1/3 octave band representation of s
    X: 1/3 octave band representation of x
    '''
    fft_size = 1024
    numBands = 17
    cf_min = 150
    frame_size = 512
    overlap = 256
    
    A, _, _ = thirdoct(20000, fft_size, numBands, cf_min)
    S = transform_matrix(s, A, frame_size, overlap, fs, fft_size=fft_size)
    X = transform_matrix(x, A, frame_size, overlap, fs, fft_size=fft_size)
    
    S = np.expand_dims(np.expand_dims(S, axis=0), axis=3)
    X = np.expand_dims(np.expand_dims(X, axis=0), axis=3)
    return S, X

def load_Network_weights(weights, dsmf=False, datasets=10):
    '''
    Load weights for the network with or without 
    
    weights: path to the file containing the trained weights
    dsmf: (bool) if True then load the DSMF's
    '''
    if dsmf:
        model = net.Network_dsfit(datasets)
    else:
        model = net.Network(datasets)
    
    # For some reason I have to start training the architecture before it accepts loading weights
    model.compile(optimizer='Adam', loss='mse')
    randx = np.random.randn(1, 100, 17, 1)
    randy = np.random.randn(1, 1)
    if dsmf:
        d = np.zeros((1,1), dtype=np.int)
        model.fit([randx, randx, d], randy, epochs=1, verbose=False)
    else:
        model.fit([randx, randx], randy, epochs=1, verbose=False)
    model.load_weights(weights)
    return model