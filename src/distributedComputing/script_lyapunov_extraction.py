# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:48:48 2015

@author: prillard
"""

import sys
import numpy as np

def lyapunov(X):
    dt = 1.0 / 200
    ndata = len(X)
    
    N2 = ndata/2
    N4 = ndata/4
    TOL = 0.000001
    
    exponent = np.zeros(N4 + 1)

    for i in range(N4, N2):
        #dist = np.min(np.abs[X - X[i]])
        #indx = np.argmin(np.abs[X - X[i]])
        dist = np.abs(X[i+1]-X[i])
        indx = i + 1
        for j in range(0, ndata-5):
            if i != j and np.abs(X[i]-X[j]) < dist:
                dist = np.abs(X[i]-X[j])
                indx = j
        #print i,indx, dist        
        expn = 0.0
        for k in range(1, 6):
            if np.abs(X[i+k] - X[indx+k]) > TOL and np.abs(X[i] - X[indx]) > TOL: 
                expn += (np.log(np.abs(X[i+k] - X[indx+k])) - np.log(np.abs(X[i] - X[indx]))) / k
                                      
        exponent[i-N4 + 1] = float(expn)/5
        
    sum_exponent = np.sum(exponent)
    return sum_exponent/((N4 + 1)*dt)
    
def compute_lyapunov_feature(X, start, stop, output_file):
    X_train_feature = []
    cpt = 0
    for line in X[start:stop]:
        print 'compute_lyapunov_feature : ', cpt + start, ' / ', stop
        res = lyapunov(line)
        X_train_feature.append(res)
        cpt += 1
    X_train_feature = np.array(X_train_feature)
    # save feature in file
    np.savetxt(output_file, X_train_feature, delimiter=";")
    

if __name__ == '__main__':
    X_path = str(sys.argv[1])
    X = np.array(np.loadtxt(X_path, delimiter=";"))
    start = int(sys.argv[2])
    stop = int(sys.argv[3])
    output_file = str(sys.argv[4])
        
    print X_path, X.shape, start, stop, output_file
    compute_lyapunov_feature(X, start, stop, output_file) 
