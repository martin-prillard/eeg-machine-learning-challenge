# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:43:08 2015

@author: prillard
"""

import script_pyeeg_extraction as pex
import script_pyrem_extraction as prex

regex_network = '.*\\.enst\\.fr'
main_dir = '/cal/homes/prillard/challenge/'
dataset_dir = '/cal/homes/prillard/challenge/features/' 
tmp_dir = '/tmp/'

X_train_label = 'X_train'
y_train_label = 'y_train'
X_test_label = 'X_test'
y_test_label = 'y_pred'

ext = '.csv'

X_train_path_tmp = tmp_dir + X_train_label + '_'
y_train_path_tmp = tmp_dir + y_train_label + '_'
X_test_path_tmp = tmp_dir + X_test_label + '_'
y_test_path_tmp = tmp_dir + y_test_label + '_'

X_train_path_ext_tmp = tmp_dir + X_train_label + ext
y_train_path_ext_tmp = tmp_dir + y_train_label + ext
X_test_path_ext_tmp = tmp_dir + X_test_label + ext
y_test_path_ext_tmp = tmp_dir + y_test_label + ext

# script
script_pyeeg_features = '/cal/homes/prillard/challenge/script_pyeeg_extraction.py'
script_pyrem_features = '/cal/homes/prillard/challenge/script_pyrem_extraction.py'
script_lyapunov_features = '/cal/homes/prillard/challenge/script_lyapunov_extraction.py'
script_get_cores = '/cal/homes/prillard/challenge/script_get_cores.py'
script_bow_local_segment_extraction = '/cal/homes/prillard/challenge/script_bow_local_segment.py'
script_bow_histogram = '/cal/homes/prillard/challenge/script_bow_histogram.py'

# filenames
# input EEGs
input_dataset = dataset_dir + 'data_challenge.mat'
input_dataset_binary = dataset_dir + 'data_challenge_binary.mat'

# BOW features
bow_X_train_segments = X_train_path_tmp + 'bow_segments.csv'
bow_X_test_segments = X_test_path_tmp + 'bow_segments.csv'
bow_cdw_file = main_dir + X_train_label + '_bow_codewords.csv'
bow_X_train_histo = X_train_path_tmp + 'bow_histo.csv'
bow_X_test_histo = X_test_path_tmp + 'bow_histo.csv'

# other features
kolmogorov_X_train = X_train_path_tmp + 'kolmogorov.csv'
kolmogorov_X_test = X_test_path_tmp + 'kolmogorov.csv'
fft_dl_X_train =  X_train_path_tmp +'fft_dl.csv'
fft_dl_X_test =  X_test_path_tmp +'fft_dl.csv'
mne_stft_X_train = X_train_path_tmp +'mne_stft.csv'
mne_stft_X_test = X_test_path_tmp +'mne_stft.csv'

features_name = {}
features_name[X_train_label]  = {pex.pyeeg_dfa             :X_train_path_tmp + pex.pyeeg_dfa + ext,
                                 pex.pyeeg_pfd             :X_train_path_tmp + pex.pyeeg_pfd + ext,
                                 pex.pyeeg_hjorth          :X_train_path_tmp + pex.pyeeg_hjorth + ext,
                                 pex.pyeeg_apen            :X_train_path_tmp + pex.pyeeg_apen + ext,
                                 pex.pyeeg_sampen          :X_train_path_tmp + pex.pyeeg_sampen + ext,
                                 pex.pyeeg_hfd             :X_train_path_tmp + pex.pyeeg_hfd + ext,
                                 pex.pyeeg_fisher_info     :X_train_path_tmp + pex.pyeeg_fisher_info + ext,
                                 pex.pyeeg_svd             :X_train_path_tmp + pex.pyeeg_svd + ext,
                                 pex.pyeeg_bin_power       :X_train_path_tmp + pex.pyeeg_bin_power + ext,
                                 pex.pyeeg_bin_power_ratio :X_train_path_tmp + pex.pyeeg_bin_power_ratio + ext,
                                 pex.pyeeg_spect           :X_train_path_tmp + pex.pyeeg_spect + ext,
                                 pex.pyeeg_hurst           :X_train_path_tmp + pex.pyeeg_hurst + ext,
                                 prex.pyrem_dfa            :X_train_path_tmp + prex.pyrem_dfa + ext,
                                 prex.pyrem_pfd            :X_train_path_tmp + prex.pyrem_pfd + ext,
                                 prex.pyrem_hjorth         :X_train_path_tmp + prex.pyrem_hjorth + ext,
                                 prex.pyrem_apen           :X_train_path_tmp + prex.pyrem_apen + ext,
                                 prex.pyrem_sampen         :X_train_path_tmp + prex.pyrem_sampen + ext,
                                 prex.pyrem_hfd            :X_train_path_tmp + prex.pyrem_hfd + ext,
                                 prex.pyrem_fisher_info    :X_train_path_tmp + prex.pyrem_fisher_info + ext,
                                 prex.pyrem_svd            :X_train_path_tmp + prex.pyrem_svd + ext,
                                 prex.pyrem_spect          :X_train_path_tmp + prex.pyrem_spect + ext,
                                 prex.pyrem_hurst          :X_train_path_tmp + prex.pyrem_hurst + ext
                                 }
features_name[X_test_label]  = {pex.pyeeg_dfa              :X_test_path_tmp + pex.pyeeg_dfa + ext,
                                 pex.pyeeg_pfd             :X_test_path_tmp + pex.pyeeg_pfd + ext,
                                 pex.pyeeg_hjorth          :X_test_path_tmp + pex.pyeeg_hjorth + ext,
                                 pex.pyeeg_apen            :X_test_path_tmp + pex.pyeeg_apen + ext,
                                 pex.pyeeg_sampen          :X_test_path_tmp + pex.pyeeg_sampen + ext,
                                 pex.pyeeg_hfd             :X_test_path_tmp + pex.pyeeg_hfd + ext,
                                 pex.pyeeg_fisher_info     :X_test_path_tmp + pex.pyeeg_fisher_info + ext,
                                 pex.pyeeg_svd             :X_test_path_tmp + pex.pyeeg_svd + ext,
                                 pex.pyeeg_bin_power       :X_test_path_tmp + pex.pyeeg_bin_power + ext,
                                 pex.pyeeg_bin_power_ratio :X_test_path_tmp + pex.pyeeg_bin_power_ratio + ext,
                                 pex.pyeeg_spect           :X_test_path_tmp + pex.pyeeg_spect + ext,
                                 pex.pyeeg_hurst           :X_test_path_tmp + pex.pyeeg_hurst + ext,
                                 prex.pyrem_dfa            :X_test_path_tmp + prex.pyrem_dfa + ext,
                                 prex.pyrem_pfd            :X_test_path_tmp + prex.pyrem_pfd + ext,
                                 prex.pyrem_hjorth         :X_test_path_tmp + prex.pyrem_hjorth + ext,
                                 prex.pyrem_apen           :X_test_path_tmp + prex.pyrem_apen + ext,
                                 prex.pyrem_sampen         :X_test_path_tmp + prex.pyrem_sampen + ext,
                                 prex.pyrem_hfd            :X_test_path_tmp + prex.pyrem_hfd + ext,
                                 prex.pyrem_fisher_info    :X_test_path_tmp + prex.pyrem_fisher_info + ext,
                                 prex.pyrem_svd            :X_test_path_tmp + prex.pyrem_svd + ext,
                                 prex.pyrem_spect          :X_test_path_tmp + prex.pyrem_spect + ext,
                                 prex.pyrem_hurst          :X_test_path_tmp + prex.pyrem_hurst + ext
                                 }

codebook_size = 200
db_family = 'db'
db_number = 4
wave_level = 100
lenght_segment = 100 #10178 / 75
local_segment_step = 20 #75 / 4 
