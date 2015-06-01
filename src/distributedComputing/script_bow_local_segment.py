#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:18:42 2015

@author: prillard
"""
import numpy as np
import pywt
import app
import sys
from numpy.fft import fft

# waves name
wave_haar = 'haar'
wave_db   = 'db8'
wave_sym  = 'sym8'
wave_coif = 'coif5'
wave_bior = 'bior2.8'
wave_rbio = 'rbio2.8'
wave_dmey = 'dmey'


###
# dwt coeffs at level x
# sqrt(sum(coeff^2)) for accurates levels (not the first one)
###
def waveCoeffs(data, wave, wave_level):
    w = pywt.wavedec(data, wavelet=wave, level=wave_level)
    res = list()    
    for i in range(1,len(w)):
        temp = 0
        for j in range(0,len(w[i])):
            temp += w[i][j]**2
        res.append(np.sqrt(temp))    
    return res

def ftt_power(data, ratio=False):
    Band=[0.5,4,8,14.5,60] #delta, theta, alpha and beta respectively
    #Band = [i for i in xrange(0, 60)]
    Fs=200
    C = fft(data)
    C = abs(C)
    Power =np.zeros(len(Band)-1);
    for Freq_Index in xrange(0,len(Band)-1):
        Freq = float(Band[Freq_Index])										
        Next_Freq = float(Band[Freq_Index+1])
        Power[Freq_Index] = sum(C[np.floor(Freq/Fs*len(data)):np.floor(Next_Freq/Fs*len(data))])
        Power_Ratio = Power/sum(Power)
    if ratio:
        return Power_Ratio
    else:
        return Power

def getSpectrum(y,Fs):
    #Gets the frequency and energy spectrum E(frq) of y(t)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    E = abs(Y)**2
    return frq, E

def computeVec(data, doNorm=1):
    Fs = 200.0;  # sampling rate
    kernel = np.hanning(80)
    kernel = kernel / kernel.sum()
    frqmax = 100
    frq0 = np.arange(0,frqmax,1)
    #print "Computing state vector"
    V = np.zeros(len(frq0))

    y = np.array(data)
    frq, E = getSpectrum(y,Fs)
    smoothed = np.convolve(kernel, E, mode='SAME')
    logeeg = np.log10(smoothed)
    logeeg0 = np.interp(frq0, frq, logeeg)
    if doNorm:
        V.append((logeeg0-logeeg0.mean())/logeeg0.std())
    else:
        V.append(logeeg0)
    return V


def compute_local_segment_extraction(X, start, stop, output_file=None):
    nb_segment = X.shape[1] / app.lenght_segment
    X_segmented = []
    for line in X[start:stop]:
        first = True
        line_temp = line
        for i in range(0, int(nb_segment/app.local_segment_step)):
            if first:
                #total_seg = np.array([waveCoeffs(segment, 'db' + str(app.wave_level), app.wave_level) for segment in np.split(line_temp, nb_segment)])
                total_seg = np.array([computeVec(segment) for segment in np.split(line_temp, nb_segment)])                
                first = False
            else:
                # get 48 segments with their 8 wavelets coeff
                #total_seg = np.concatenate((total_seg, np.array([waveCoeffs(segment, 'db' + str(app.wave_level), app.wave_level) for segment in np.split(line_temp, nb_segment)])), axis=0)
                total_seg = np.concatenate((total_seg, np.array([computeVec(segment) for segment in np.split(line_temp, nb_segment)])), axis=0)
            line_temp = np.roll(line_temp, app.local_segment_step)
        X_segmented.append(np.matrix(total_seg).A1)
    X_segmented = np.array(X_segmented)
    # concat file
    if output_file:
        np.savetxt(output_file, X_segmented, fmt='%.8e', delimiter=";")
        print 'compute_local_segment_extraction : ', output_file, '[', start, ':', stop, ']', X_segmented.shape
    return X_segmented 


if __name__ == '__main__':
    X_path = str(sys.argv[1])
    X = np.array(np.loadtxt(X_path, delimiter=";"))
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    output_file = str(sys.argv[4])
    print X_path, X.shape, start, stop, output_file
    compute_local_segment_extraction(X, start, stop, output_file)