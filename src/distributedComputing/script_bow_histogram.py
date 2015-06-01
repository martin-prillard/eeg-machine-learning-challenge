# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:29:00 2015

@author: prillard
"""
import time
from itertools import groupby
import numpy as np
import sys
import app
from sklearn.neighbors import NearestNeighbors
import utils

def compute_histogram_features(X, start, stop, output_file):
    t0 = time.time()
    # apply nearest neighbors model
    codebook = np.array(np.loadtxt(app.bow_cdw_file, delimiter=";"))
    nbrs = NearestNeighbors(n_neighbors=app.codebook_size, algorithm='auto').fit(codebook)
    X_bagow = []
    cpt = 0
    for sample in X[start:stop]:
        print 'compute_histogram_features : ', cpt
        codewords = []
        # compute all the nearest codeword (centroid) for each sample
        distances, indices = nbrs.kneighbors(sample)
        for indice in indices:
		    # return the nearest codeword (centroid) index
            codewords.append(int(indice[0]))
        # order the list
        codewords = sorted(codewords, key=int)
        # wordcount on codewords
        histogram = {}
        key_old = -1
        for key, group in groupby(codewords):
            histogram[key] = len(list(group))
            # if step
            if (key > key_old+1):
                # init to zero
                diff = (key-1) - key_old
                for i in range(0, diff):
                    histogram[key_old+i+1] = 0
            key_old = key
            
        # dict to array
        histogram_list = []
        for key, value in histogram.iteritems():
            histogram_list.append(value)
        # fit with zero values if needed
        while (len(histogram_list) < app.codebook_size):
            histogram_list.append(0)
            
        X_bagow.append(np.array(histogram_list))
        cpt += 1
    X_bagow = np.array(X_bagow)
    
    # concat file
    if output_file:
        np.savetxt(output_file, X_bagow, fmt='%.8e', delimiter=";")
    print 'END [', start, ':', stop, '] : ', time.time() - t0
    return X_bagow 
    
    
if __name__ == '__main__':
    X_path = str(sys.argv[1])
    X = utils.array_segments_to_matrix(np.array(np.loadtxt(X_path, delimiter=";")), app.wave_level)
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    output_file = str(sys.argv[4])
    print X_path, X.shape, start, stop, output_file
    compute_histogram_features(X, start, stop, output_file)
