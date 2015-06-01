# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:40:46 2015

@author: prillard
"""

import app
import numpy as np
from scipy import io
import random
import scipy.io as sio
import math

def array_segments_to_matrix(X, chunk_size):
    return np.array([X_segment.reshape(len(X_segment) / chunk_size, chunk_size) for X_segment in X])

def eeg_to_binary_value():
    dataset = io.loadmat(app.input_dataset)
    X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']
    X_binary = {}
    X_binary['X_train'] = []
    for line in X_train:
        threshold = np.median(np.abs(line))
        line_binary = []
        for element in line:
            if np.abs(element) > threshold:
                line_binary.append(1)
            else:
                line_binary.append(0)
        X_binary['X_train'].append(line_binary)
    X_binary['X_test'] = []
    for line in X_test:
        threshold = np.median(np.abs(line))
        line_binary = []
        for element in line:
            if np.abs(element) > threshold:
                line_binary.append(1)
            else:
                line_binary.append(0)
        X_binary['X_test'].append(line_binary)
    sio.savemat(app.input_dataset_binary, X_binary)
    print 'file save : ', app.input_dataset_binary

def write_dataset_elements(lengh=None):
    dataset = io.loadmat(app.input_dataset)
    dataset['X_train'].shape
    X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']
    X_train.shape, y_train.shape, X_test.shape
    
    if lengh == None:
        # full dataset
        np.savetxt(app.X_train_path_ext_tmp, X_train, fmt='%.8e', delimiter=";")
        np.savetxt(app.y_train_path_ext_tmp, y_train, fmt='%s', delimiter=";")
        np.savetxt(app.X_test_path_ext_tmp, X_test, fmt='%.8e', delimiter=";")
        print X_train.shape, y_train.shape, X_test.shape
    else:
        # random sample dataset
        X_train_min = []
        y_train_min = []
        rdm = random.sample(xrange(len(X_train)), lengh)
        for i in rdm:
            X_train_min.append(X_train[i])
            y_train_min.append(y_train[i])
        X_train_min = np.array(X_train_min)
        y_train_min = np.array(y_train_min)
        np.savetxt(app.X_train_path_tmp, X_train_min, fmt='%.8e', delimiter=";")
        np.savetxt(app.y_train_path_tmp, y_train_min, fmt='%s', delimiter=";")
        print X_train_min.shape, y_train_min.shape
        
def replace_nan_by_mean(XX_train, XX_test):
    for i in range(1, XX_train.shape[1]):
        XX_train_copy  = XX_train.T[len(XX_train.T)-i][~np.isnan(XX_train.T[len(XX_train.T)-i])]
        mean_feature = float(np.mean(XX_train_copy))
        n=0
        for value in XX_train.T[len(XX_train.T)-i]:
            if math.isnan(value):
                XX_train.T[len(XX_train.T)-i][n] = mean_feature
            n+=1
    
    for i in range(1, XX_test.shape[1]):
        XX_test_copy = XX_test.T[len(XX_test.T)-i][~np.isnan(XX_test.T[len(XX_test.T)-i])]
        mean_feature = float(np.mean(XX_test_copy))
        n=0
        for value in XX_test.T[len(XX_test.T)-i]:
            if math.isnan(value):
                XX_test.T[len(XX_test.T)-i][n] = mean_feature
            n+=1
    return XX_train, XX_test