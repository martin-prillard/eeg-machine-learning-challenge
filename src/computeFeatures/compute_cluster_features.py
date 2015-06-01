# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:47:48 2015

@author: prillard
"""

import app
import script_pyeeg_extraction as pex
import script_pyrem_extraction as prex
import script_lyapunov_extraction as lex
import cluster_util
import numpy as np
import utils
import scipy.io as sio


##########################################################################################
# TESTS
##########################################################################################

#cluster = cluster_util.get_cluster(regex_network, 999999)
cluster = cluster_util.get_cluster_classrom([19,39,7], 1, 39, 'c130')
#print len(cluster), cluster
#utils.write_dataset_elements()

##########################################################################################
# PyEEG features
##########################################################################################

# DFA : Detrended fluctuation analysis
pyeeg_feature = pex.pyeeg_dfa
other_args = {pex.var_pyeeg_feature:pyeeg_feature}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_dfa], other_args)
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_dfa], other_args)

# PFD : Petrosian Fractal Dimension
pyeeg_feature = pex.pyeeg_pfd
other_args = {pex.var_pyeeg_feature:pyeeg_feature}
for el in other_args:
    print el, other_args[el]
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_pfd], other_args)
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_pfd], other_args)

# hjorth : Hjorth mobility and complexity of a time series
pyeeg_feature = pex.pyeeg_hjorth
other_args = {pex.var_pyeeg_feature:pyeeg_feature}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_hjorth], other_args)
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_hjorth], other_args)

# ApEn : Approximate entropy
pyeeg_feature = pex.pyeeg_apen
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_dim:10}
X_train = np.array(np.loadtxt(app.X_train_path_ext_tmp, delimiter=";"))
other_args[pex.var_R] = np.std(X_train) * 0.3
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_apen], other_args)
X_test = np.array(np.loadtxt(app.X_test_path_ext_tmp, delimiter=";"))
other_args[pex.var_R] = np.std(X_test) * 0.3
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_apen], other_args)

# SampEn : sample entropy (SampEn) of series X, specified by M and R (very close to ApEn)
pyeeg_feature = pex.pyeeg_sampen
other_args = {}
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_dim:10}
X_train = np.array(np.loadtxt(app.X_train_path_ext_tmp, delimiter=";"))
other_args[pex.var_R] = np.std(X_train) * 0.3
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_sampen], other_args) 
X_test = np.array(np.loadtxt(app.X_test_path_ext_tmp, delimiter=";"))
other_args[pex.var_R] = np.std(X_test) * 0.3
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_sampen], other_args) 

# hfd : Higuchi Fractal Dimension of a time series X, kmax is an HFD parameter
pyeeg_feature = pex.pyeeg_hfd
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_Kmax:5}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_hfd], other_args)
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_hfd], other_args)

# fisher_info : Fisher information of a time series
pyeeg_feature = pex.pyeeg_fisher_info
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_dim:10, pex.var_TAU:4}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_fisher_info], other_args) 
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_fisher_info], other_args) 

# SVD : SVD Entropy
pyeeg_feature = pex.pyeeg_svd
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_dim:10, pex.var_TAU:4}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_svd], other_args)     
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_svd], other_args)     

# bin power : power in each frequency bin specified by Band from FFT result of X (default : X is a real signal). 
# Power Spectral Density (PSD), spectrum power in a set of frequency bins, and, Relative Intensity Ratio (RIR), PSD normalized by total power in all frequency bins
pyeeg_feature = pex.pyeeg_bin_power
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_Band:','.join([str(2*i+1) for i in xrange(0, 43)]), pex.var_Fs:200} # 0.5~85 Hz
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_bin_power], other_args) 
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_bin_power], other_args) 

# bin power ratio
pyeeg_feature = pex.pyeeg_bin_power_ratio
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_Band:','.join([str(2*i+1) for i in xrange(0, 43)]), pex.var_Fs:200}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_bin_power_ratio], other_args)    
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_bin_power_ratio], other_args)    

# spect : spectral entropy of a time series
pyeeg_feature = pex.pyeeg_spect
other_args = {pex.var_pyeeg_feature:pyeeg_feature, pex.var_Band:','.join([str(2*i+1) for i in xrange(0, 43)]), pex.var_Fs:200}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_spect], other_args)    
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_spect], other_args)    

# hurst : Hurst exponent of X
pyeeg_feature = pex.pyeeg_hurst
other_args = {pex.var_pyeeg_feature:pyeeg_feature}
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][pex.pyeeg_hurst], other_args)   
cluster_util.apply_cores_function(app.script_pyeeg_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][pex.pyeeg_hurst], other_args)   


##########################################################################################
# Pyrem features
##########################################################################################

# DFA : Detrended fluctuation analysis
pyrem_feature = prex.pyrem_dfa
other_args = {prex.var_pyrem_feature:pyrem_feature}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_dfa], other_args)  
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_dfa], other_args)

# PFD : Petrosian Fractal Dimension
pyrem_feature = prex.pyrem_pfd
other_args = {prex.var_pyrem_feature:pyrem_feature}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_pfd], other_args)  
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_pfd], other_args)

# hjorth : Hjorth mobility and complexity of a time series
pyrem_feature = prex.pyrem_hjorth
other_args = {prex.var_pyrem_feature:pyrem_feature}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_hjorth], other_args)  
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_hjorth], other_args)  

# ApEn : Approximate entropy
pyrem_feature = prex.pyrem_apen
other_args = {prex.var_pyrem_feature:pyrem_feature, prex.var_m:5, prex.var_R:1.5}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_apen], other_args)   
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_apen], other_args)   

# SampEn : sample entropy (SampEn) of series X, specified by M and R (very close to ApEn)
pyrem_feature = prex.pyrem_sampen
other_args = {prex.var_pyrem_feature:pyrem_feature, prex.var_m:10, prex.var_R:1.5}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_sampen], other_args)   
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_sampen], other_args)   

# hfd : Higuchi Fractal Dimension of a time series X, kmax is an HFD parameter
# TODO

# fisher_info : Fisher information of a time series
# TODO

# SVD : SVD Entropy
# TODO

# spect : spectral entropy of a time series
# TODO

# hurst : Hurst exponent of X
pyrem_feature = prex.pyrem_hurst
other_args = {prex.var_pyrem_feature:pyrem_feature}
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_train_path_ext_tmp, cluster, app.features_name[app.X_train_label][prex.pyrem_hurst], other_args)   
cluster_util.apply_cores_function(app.script_pyrem_features, app.X_test_path_ext_tmp, cluster, app.features_name[app.X_test_label][prex.pyrem_hurst], other_args)   