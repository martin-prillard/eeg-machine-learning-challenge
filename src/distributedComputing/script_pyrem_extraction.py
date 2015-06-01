# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:02:33 2015

@author: prillard
"""

import sys
import numpy as np
from pyrem_univariate import *

pyrem_label           = 'pyrem'
# Pyrem features
pyrem_dfa             = pyrem_label + '_dfa'
pyrem_pfd             = pyrem_label + '_pfd'
pyrem_hjorth          = pyrem_label + '_hjorth'
pyrem_apen            = pyrem_label + '_apen'
pyrem_sampen          = pyrem_label + '_sampen'
pyrem_hfd             = pyrem_label + '_hfd'
pyrem_fisher_info     = pyrem_label + '_fisher_info'
pyrem_svd             = pyrem_label + '_svd'
pyrem_spect           = pyrem_label + '_spect'
pyrem_hurst           = pyrem_label + '_hurst'

var_pyrem_feature = 'pyrem_feature'
var_m = 'm'
var_R = 'R'
var_Kmax = 'Kmax'
var_TAU = 'TAU'
var_Band = 'Band'
var_de = 'de'
            
def apply_pyrem(line, other_args):
    
    if var_pyrem_feature in other_args:
        pyrem_feature = other_args[var_pyrem_feature]
    if var_m in other_args:
        m = int(other_args[var_m])
    if var_R in other_args:
        R = float(other_args[var_R])
    if var_Kmax in other_args:
        Kmax = int(other_args[var_Kmax])
    if var_TAU in other_args:
        TAU = int(other_args[var_TAU])
    if var_Band in other_args:
        Band = np.array(other_args[var_Band].split(','))
    if var_de in other_args:
        de = np.array(other_args[var_de].split(','))
        
    return {
          pyrem_dfa             : lambda x: dfa(line),
          pyrem_pfd             : lambda x: pfd(line),
          pyrem_hjorth          : lambda x: hjorth(line),
          pyrem_apen            : lambda x: ap_entropy(line, m, R),
          pyrem_sampen          : lambda x: samp_entropy(line, m, R),
          pyrem_hfd             : lambda x: hfd(line, Kmax),
          pyrem_fisher_info     : lambda x: fisher_info(line, TAU, de),
          pyrem_svd             : lambda x: svd_entropy(line, TAU, de),
          pyrem_spect           : lambda x: spectral_entropy(line, Band),
          pyrem_hurst           : lambda x: hurst(line),
    }[pyrem_feature](1)
    
    
def compute_pyrem_feature(X, start, stop, output_file, other_args):
    X_train_feature = []
    cpt = 0
    for line in X[start:stop]:
        res = apply_pyrem(line, other_args)
        X_train_feature.append(res)
        cpt += 1
    X_train_feature = np.array(X_train_feature)
    # save feature in file
    np.savetxt(output_file, X_train_feature, delimiter=";")
    print 'compute_pyrem_feature : ', output_file


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
    compute_pyrem_feature(X, start, stop, output_file, other_args) 
