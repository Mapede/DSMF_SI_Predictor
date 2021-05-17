# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:23:54 2021

@author: mbp
"""
import numpy as np
#from functools import partial
#import matplotlib.pyplot as plt
#from scipy.io import loadmat, savemat
#from scipy.stats import kendalltau
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Reshape

epsilon = 1e-12

@tf.function
def custom_std(x, axis=-1, keepdims=False):
    """
    tf's std caused instabillity during training.
    Assumes mean already removed!
    """
    return tf.math.sqrt(tf.math.reduce_mean(x*x, axis=axis, keepdims=keepdims))

@tf.function
def ECorr(x):
    mu = tf.math.reduce_mean(x, axis=2, keepdims=True)
    y = x - mu
    sigma = np.sqrt(y.shape[2]) * custom_std(y,axis=2, keepdims=True)
    y = y / (sigma + epsilon)
    
    mu = tf.math.reduce_mean(y, axis=3, keepdims=True)
    z = y - mu
    sigma = np.sqrt(z.shape[3]) * custom_std(z,axis=3, keepdims=True)
    z = z / (sigma + epsilon)
    return z

@tf.function
def EstoiCorrelation(x, y): 
    """ Compute normalzied correlation across time and frequency
    x, y are (batch, timeframes, time, freq, 1) """
    x_norm = ECorr(x)
    y_norm = ECorr(y)
#    x_norm = tf.reshape(x_norm, (x_norm.shape[1], -1, *x_norm.shape[4:]))
#    y_norm = tf.reshape(y_norm, (y_norm.shape[1], -1, *y_norm.shape[4:]))
    corrs = tf.math.reduce_sum(x_norm * y_norm, axis=(2,3)) / x.shape[2]
    return corrs

@tf.function
def SlidingCorrelation(x, y, corr, mean_output=False):
    """ Compute normalzied correlation across time or frequency within a sliding window.
    x, y are (batch, time, freq, 1) """
    frame_x = tf.signal.frame(x, frame_length=30, frame_step=1, axis=1)
    frame_y = tf.signal.frame(y, frame_length=30, frame_step=1, axis=1)
    correlation = corr(frame_x, frame_y)
#    print(x.shape, frame_x.shape, correlation.shape)
    if mean_output:
        correlation = tf.math.reduce_mean(correlation, axis=2, keepdims=True)
    return correlation


class Network_dsfit(tf.keras.Model):
    """ 
    Architecture used for training with DSMF
    Inputs = S, X, d
    
    datasets: number of unique DSMF's
    K: number of kernels
    stride: convolutional stride
    L: number of Conv layers
    """
    def __init__(self, datasets, K=20, stride=1, L=3):
        super(Network_dsfit, self).__init__()
        self.datasets = datasets
        self.L = L
        self.conv = []
        
        for l in range(L):
            self.conv.append(Conv2D(K, (3,3), strides=(stride,stride), padding='same', activation='relu', data_format='channels_last'))
            
        self.Reshape = Reshape((-1,17*K))
        self.Correlation = EstoiCorrelation
        self.SlidingCorrelation = SlidingCorrelation
        self.Sig = Dense(units=self.datasets, input_shape=(1,), activation='sigmoid', kernel_constraint='NonNeg')
        
    def call(self, inputs):
        s, x, d = inputs
        
        S = s
        for l in range(self.L):
            S = self.conv[l](S)
        S = self.Reshape(S)
        
        X = x
        for l in range(self.L):
            X = self.conv[l](X)
        X = self.Reshape(X)
        
        T = self.SlidingCorrelation(S, X, self.Correlation, False)
        T = tf.reduce_mean(T, axis=1, keepdims=True)
        
        # compute DSMF's
        T = self.Sig(T)
        T = tf.one_hot(d, self.datasets) * tf.expand_dims(T,1)
        T = tf.math.reduce_sum(T, axis=2)
        return T
    
class Network(tf.keras.Model):
    """ 
    Architecture used for testing (no DSMF)
    Inputs = S, X
    
    datasets: number of unique DSMF's (only used when loading weights trained with Network_dsfit)
    K: number of kernels
    stride: convolutional stride
    L: number of Conv layers
    """
    def __init__(self, datasets, K=20, stride=1, L=3):
        super(Network, self).__init__()
        self.datasets = datasets
        self.L = L
        self.conv = []
        
        for l in range(L):
            self.conv.append(Conv2D(K, (3,3), strides=(stride,stride), padding='same', activation='relu', data_format='channels_last'))
            
        self.Reshape = Reshape((-1,17*K))
        self.Correlation = EstoiCorrelation
        self.SlidingCorrelation = SlidingCorrelation
        self.Sig = Dense(units=datasets, input_shape=(1,), activation='sigmoid', kernel_constraint='NonNeg')
        
    def call(self, inputs):
        s, x = inputs
        
        S = s
        for l in range(self.L):
            S = self.conv[l](S)
        S = self.Reshape(S)
        
        X = x
        for l in range(self.L):
            X = self.conv[l](X)
        X = self.Reshape(X)
        
        T = self.SlidingCorrelation(S, X, self.Correlation, False)
        T = tf.reduce_mean(T, axis=1, keepdims=True)
        
        # This is unused, but makes it easier to load weights trained with DSMF
        U = self.Sig(T)
        U = tf.one_hot(np.zeros((1,self.datasets),dtype=np.int), self.datasets) * tf.expand_dims(U,1)
        U = tf.math.reduce_sum(U, axis=2)
        return T