# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:20:59 2015

@author: prillard
"""

import sys
import numpy as np
from pyeeg import *

pyeeg_label           = 'pyeeg'
# PyEEG features
pyeeg_dfa             = pyeeg_label + '_dfa'
pyeeg_pfd             = pyeeg_label + '_pfd'
pyeeg_hjorth          = pyeeg_label + '_hjorth'
pyeeg_apen            = pyeeg_label + '_apen'
pyeeg_sampen          = pyeeg_label + '_sampen'
pyeeg_hfd             = pyeeg_label + '_hfd'
pyeeg_fisher_info     = pyeeg_label + '_fisher_info'
pyeeg_svd             = pyeeg_label + '_svd'
pyeeg_bin_power       = pyeeg_label + '_bin_power'
pyeeg_bin_power_ratio = pyeeg_label + '_bin_power_ratio'
pyeeg_spect           = pyeeg_label + '_spect'
pyeeg_hurst           = pyeeg_label + '_hurst'

var_pyeeg_feature = 'pyeeg_feature'
var_dim = 'DIM'
var_R = 'R'
var_Kmax = 'Kmax'
var_TAU = 'TAU'
var_Band = 'Band'
var_Fs = 'Fs'
    
def get_W(line, TAU, DIM):
    M = embed_seq(line, TAU, DIM)
    W = svd(M, compute_uv=0)
    W /= sum(W)
    return W
            
def apply_pyeeg(line, other_args):
    
    if var_pyeeg_feature in other_args:
        pyeeg_feature = other_args[var_pyeeg_feature]
    if var_dim in other_args:
        DIM = int(other_args[var_dim])
    if var_R in other_args:
        R = float(other_args[var_R])
    if var_Kmax in other_args:
        Kmax = int(other_args[var_Kmax])
    if var_TAU in other_args:
        TAU = int(other_args[var_TAU])
    if var_Band in other_args:
        Band = np.array(other_args[var_Band].split(','))
    if var_Fs in other_args:
        Fs = int(other_args[var_Fs])   
        
    return {
          pyeeg_dfa             : lambda x: dfa(line),
          pyeeg_pfd             : lambda x: pfd(line),
          pyeeg_hjorth          : lambda x: hjorth(line),
          pyeeg_apen            : lambda x: ap_entropy(line, DIM, R),
          pyeeg_sampen          : lambda x: samp_entropy(line, DIM, R),
          pyeeg_hfd             : lambda x: hfd(line, Kmax),
          pyeeg_fisher_info     : lambda x: fisher_info(line, TAU, DIM, get_W(line, TAU, DIM)),
          pyeeg_svd             : lambda x: svd_entropy(line, TAU, DIM, get_W(line, TAU, DIM)),
          pyeeg_bin_power       : lambda x: bin_power(line, Band, Fs)[0],
          pyeeg_bin_power_ratio : lambda x: bin_power(line, Band, Fs)[1],
          pyeeg_spect           : lambda x: spectral_entropy(line, Band, Fs),
          pyeeg_hurst           : lambda x: hurst(line),
    }[pyeeg_feature](1)

def compute_pyeeg_feature(X, start, stop, output_file, other_args):
    X_train_feature = []
    cpt = 0
    for line in X[start:stop]:
        print 'compute_pyeeg_feature : ', cpt + start, ' / ', stop
        res = apply_pyeeg(line, other_args)
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
    
    other_args = {}
    # other_args
    for arg in sys.argv[5:len(sys.argv)]:
        arg_2 = arg.split('=')
        other_args[arg_2[0]] = arg_2[1]
        
    print X_path, X.shape, start, stop, output_file, other_args
    compute_pyeeg_feature(X, start, stop, output_file, other_args) 
