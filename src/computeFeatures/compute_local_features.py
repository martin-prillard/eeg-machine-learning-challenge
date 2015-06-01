# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 14:28:25 2015

@author: prillard
"""

from numpy.fft import fft
import scipy.io as io
import numpy as np
import app
from gensim import similarities
import zlib 
import beta_ntf as bntf
import utils
import mne
import pywt
import stats
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

##########################################################################################
# TESTS
##########################################################################################
utils.write_dataset_elements()
dataset = io.loadmat(app.input_dataset)
X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']

##########################################################################################
# Wavelets
##########################################################################################
# waves name
wave_haar = 'haar'
wave_db   = 'db4'
wave_sym  = 'sym4'
wave_coif = 'coif4'
wave_bior = 'bior2.4'
wave_rbio = 'rbio2.4'
wave_dmey = 'dmey'
# levels
levels=12

def wavelist():
    for w in pywt.families():
        print pywt.wavelist(w)    

###
# dwt coeffs at level x
# sqrt(sum(coeff^2)) for accurates levels (not the first one)
###
def waveCoeffs(data,wave='db4',levels=5):
    w = pywt.wavedec(data,wavelet=wave,level=levels)
    res_RSE = list()
    res_kurtosis = list()
    res_std = list()
    for i in range(1,len(w)):
        temp = 0
        for j in range(0,len(w[i])):
            temp += w[i][j]**2
        res_RSE.append(np.sqrt(temp))    
        res_kurtosis.append(stats.kurtosis(w[i]))
        res_std.append(np.std(w[i]))
    return res_RSE, res_kurtosis, res_std

def compute_wavelet_file(X, X_label, wave_feature, levels):
    X_wavelet_RSE = []
    X_wavelet_kurtosis = []
    X_wavelet_std = []
    for line in X:
        res_RSE, res_kurtosis, res_std = waveCoeffs(line, wave_feature, levels)
        X_wavelet_RSE.append(res_RSE) 
        X_wavelet_kurtosis.append(res_kurtosis) 
        X_wavelet_std.append(res_std) 
    # save features in files
    np.savetxt(X_label + '_wavelet_' + wave_feature + '_RSE.csv', np.array(X_wavelet_RSE), delimiter=";")
    np.savetxt(X_label + '_wavelet_' + wave_feature + '_kurtosis.csv', np.array(X_wavelet_kurtosis), delimiter=";")
    np.savetxt(X_label + '_wavelet_' + wave_feature + '_std.csv', np.array(X_wavelet_std), delimiter=";")
    
def get_wavelet_feature_file(X_label, wave_feature, kind):
        return {
          wave_haar : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_haar + '_' + kind + '.csv', delimiter=";")),
          wave_db   : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_db + '_' + kind + '.csv', delimiter=";")),
          wave_sym  : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_sym + '_' + kind + '.csv', delimiter=";")),
          wave_coif : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_coif + '_' + kind + '.csv', delimiter=";")),
          wave_bior : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_bior + '_' + kind + '.csv', delimiter=";")),
          wave_rbio : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_rbio + '_' + kind + '.csv', delimiter=";")),
          wave_dmey : lambda x: np.array(np.loadtxt(X_label + '_wavelet_' + wave_dmey + '_' + kind + '.csv', delimiter=";")),
    }[wave_feature](1)

def compute_all_wavelets():
   Pipeline([
      ('wavelets_features', FeatureUnion([
        ('haar_train', compute_wavelet_file(X_train, 'X_train', wave_haar, levels)),
        ('haar_test', compute_wavelet_file(app.X_test, 'X_test', wave_haar, levels)),
        ('db_train', compute_wavelet_file(app.X_train, 'X_train', wave_db, levels)),
        ('db_test', compute_wavelet_file(app.X_test, 'X_test', wave_db, levels)),
        ('sym_train', compute_wavelet_file(app.X_train, 'X_train', wave_sym, levels)),
        ('sym_test', compute_wavelet_file(app.X_test, 'X_test', wave_sym, levels)),
        ('coif_train', compute_wavelet_file(app.X_train, 'X_train', wave_coif, levels)),
        ('coif_test', compute_wavelet_file(app.X_test, 'X_test', wave_coif, levels)),
        ('bior_train', compute_wavelet_file(app.X_train, 'X_train', wave_bior, levels)),
        ('bior_test', compute_wavelet_file(app.X_test, 'X_test', wave_bior, levels)),
        ('rbio_train', compute_wavelet_file(app.X_train, 'X_train', wave_rbio, levels)),
        ('rbio_test', compute_wavelet_file(app.X_test, 'X_test', wave_rbio, levels)),
        ('dmey_train', compute_wavelet_file(app.X_train, 'X_train', wave_dmey, levels)),
        ('dmey_test', compute_wavelet_file(app.X_test, 'X_test', wave_dmey, levels))
      ]))
    ])   
    
##########################################################################################
# FFT (by David Luz)
##########################################################################################

def getSpectrum(y,Fs):
    """
    Gets the frequency and energy spectrum E(frq) of y(t)
    """
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    E = abs(Y)**2
    return frq, E

def computeVec(X_train,indices,frq0,doNorm,kernel):
    Fs = 200.0;  # sampling rate
    #print "Computing state vector"
    V = np.zeros((len(indices),len(frq0)))
    m,n = V.shape
    icnt = 0
    for i in indices:
        #print i
        y = X_train[i]
        frq, E = getSpectrum(y,Fs)
        smoothed = np.convolve(kernel, E, mode='SAME')
        logeeg = np.log10(smoothed)
        logeeg0 = np.interp(frq0, frq, logeeg)
        if doNorm:
            V[icnt,:] = (logeeg0-logeeg0.mean())/logeeg0.std()
        else:
            V[icnt,:] = logeeg0
        icnt += 1
    return V
    
kernel = np.hanning(80)
kernel = kernel / kernel.sum()

frqmax = 100
frq0 = np.arange(0,frqmax,1)
doNorm = 1

L  = len(X_train)
L_t = len(X_test)

indx  = np.arange(0,L)
indx_t = np.arange(0,L_t)

V_train = computeVec(X_train,indx,frq0,doNorm, kernel)
V_test = computeVec(X_test,indx_t,frq0,doNorm, kernel)

np.savetxt(app.fft_dl_X_train, V_train, fmt='%.8e', delimiter=";")
np.savetxt(app.fft_dl_X_test, V_test, fmt='%.8e', delimiter=";")

##########################################################################################
# KOLMOGOROV features
##########################################################################################

dataset = io.loadmat(app.input_dataset)
X_train_binary, X_test_binary = dataset['X_train'], dataset['X_test']
 
def kolmogorov(s):
  l = float(len(s))
  compr = zlib.compress(s)
  c = float(len(compr))
  return c/l 
  
# X_train
kolmogorov_feature = []
for line in X_train_binary:
    kolmogorov_feature.append(kolmogorov(np.array(line)))
kolmogorov_feature = np.array(kolmogorov_feature)
np.savetxt(app.kolmogorov_X_train, kolmogorov_feature, fmt='%.8e', delimiter=";")

# X_test
kolmogorov_feature = []
for line in X_test_binary:
    kolmogorov_feature.append(kolmogorov(np.array(line)))
kolmogorov_feature = np.array(kolmogorov_feature)
np.savetxt(app.kolmogorov_X_test, kolmogorov_feature, fmt='%.8e', delimiter=";")


##########################################################################################
# NMF features
##########################################################################################
X_train_stft = np.sum(np.abs(mne.time_frequency.stft(X_train, 200)), axis=2)
np.savetxt(app.mne_stft_X_train, X_train_stft, fmt='%.8e', delimiter=";")

X_test_stft = np.sum(np.abs(mne.time_frequency.stft(X_test, 200)), axis=2)
np.savetxt(app.mne_stft_X_test, X_test_stft, fmt='%.8e', delimiter=";")
##########################################################################################
# NMF features
##########################################################################################


X = np.abs(mne.time_frequency.stft(X_train, 200))
X2 = []
for i in range(0, len(X_train)):
    X2.append(X[i].ravel())    
X2 = np.array(X2)

# induced_power
# TODO
"""
import numpy as np
from mne.time_frequency import induced_power


Fs = 200
frequencies = np.arange(7, 30, 3) 
n_cycles = 2
decim = 3
X2 = np.zeros((10178, 1, 6000))
X2[:,0,:] = X_train

for line in X_train[0:5]:
    power, phase_lock = induced_power(line, Fs=Fs, frequencies=frequencies,
                                      n_cycles=n_cycles, n_jobs=1, use_fft=False,
                                      decim=decim, zero_mean=True)
"""                                      
##########################################################################################
# beta_NTF features
##########################################################################################
# TODO
"""
model = bntf.build_model(X2, n_components=200, n_iter=10, beta=0)
print type(model), model.shape
XX_train_min = model[:,0:200]
y_train_min = y_train

model2 = parafac(beta_ntf.factors_)

import pandas as pd
import numpy as np
import cluster_util
import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import app
import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix

grid = GridSearchCV(RandomForestClassifier(), param_grid={}, cv=3)
grid.fit(XX_train_min, y_train_min)
print 'RandomForestClassifier : ', grid.grid_scores_
scores = [x[1] for x in grid.grid_scores_]
print scores
"""
##########################################################################################
# MatLab features
##########################################################################################

# Largest lyapunov exponent of time series with Rosenstein's Algorithm :
# SEE compute_lyarosenstein.m

# Largest  :
# SEE compute_LZ_exhaustive_complexity.m
# SEE compute_LZ_primitive_complexity.m


