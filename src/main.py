# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:01:06 2015

@author: prillard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io, stats, misc
import pandas as pd
from math import log
from itertools import groupby
from random import *
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, scale
from sklearn import gaussian_process
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.svm import SVC
import mne
from sklearn.cluster import KMeans
import random
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
from pandas import Series
from sklearn import preprocessing
import math
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
import operator
import pywt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import check_random_state
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from math import log
from random import *
from sklearn.cluster import KMeans
import random
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import math
from sklearn.cluster import MiniBatchKMeans
import pywt
from time import time
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log
from itertools import groupby
from random import *
from sklearn.preprocessing import StandardScaler, scale
from sklearn import gaussian_process
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.svm import SVC
import random
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import FeatureUnion
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
from pandas import Series
from sklearn import preprocessing
import math
import numpy as np
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
import pywt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.cross_validation import ShuffleSplit
from sklearn import cross_validation
from sklearn.utils import check_random_state
from sklearn import datasets
import thread
import app
import utils
from scipy.spatial.distance import cdist


##########################################################################################
# load dataset
##########################################################################################

# load dataset
dataset = io.loadmat(app.input_dataset)
dataset['X_train'].shape
X_train, y_train, X_test = dataset['X_train'], dataset['y_train'], dataset['X_test']
print dataset.keys()
print X_train.shape, y_train.shape, X_test.shape
labels = np.unique(y_train)
print(labels)

##########################################################################################
# Features merging
##########################################################################################

# X_train
X_train_bow_histo = 'X_train_bow_histo'
X_train_pyeeg_dfa = 'X_train_pyeeg_dfa'
X_train_pyeeg_pfd = 'X_train_pyeeg_pfd'
X_train_pyeeg_hjorth = 'X_train_pyeeg_hjorth'
X_train_pyeeg_apen = 'X_train_pyeeg_apen'
X_train_pyeeg_sampen = 'X_train_pyeeg_sampen'
X_train_pyeeg_hfd = 'X_train_pyeeg_hfd'
X_train_pyeeg_fisher_info = 'X_train_pyeeg_fisher_info'
X_train_pyeeg_svd = 'X_train_pyeeg_svd'
X_train_pyeeg_bin_power = 'X_train_pyeeg_bin_power'
X_train_pyeeg_bin_power_ratio = 'X_train_pyeeg_bin_power_ratio'
X_train_pyeeg_spect = 'X_train_pyeeg_spect'
X_train_pyeeg_hurst = 'X_train_pyeeg_hurst'
X_train_pyrem_sampen = 'X_train_pyrem_sampen'
X_train_pyrem_apen = 'X_train_pyrem_apen'
X_train_pyrem_sampen = 'X_train_pyrem_sampen'
X_train_pyrem_dfa = 'X_train_pyrem_dfa'
X_train_pyrem_hjorth = 'X_train_pyrem_hjorth'
X_train_pyrem_pfd = 'X_train_pyrem_pfd'
X_train_pyrem_hurst = 'X_train_pyrem_hurst'
X_train_kolmogorov = 'X_train_kolmogorov'
X_train_wavelet_haar_rse = 'X_train_wavelet_haar_rse'
X_train_wavelet_db_rse = 'X_train_wavelet_db_rse'
X_train_wavelet_sym_rse = 'X_train_wavelet_sym_rse'
X_train_wavelet_coif_rse = 'X_train_wavelet_coif_rse'
X_train_wavelet_bior_rse = 'X_train_wavelet_bior_rse'
X_train_wavelet_rbio_rse = 'X_train_wavelet_rbio_rse'
X_train_wavelet_dmey_rse = 'X_train_wavelet_dmey_rse'
X_train_wavelet_haar_kurtosis = 'X_train_wavelet_haar_kurtosis'
X_train_wavelet_db_kurtosis = 'X_train_wavelet_db_kurtosis'
X_train_wavelet_sym_kurtosis = 'X_train_wavelet_sym_kurtosis'
X_train_wavelet_coif_kurtosis = 'X_train_wavelet_coif_kurtosis'
X_train_wavelet_bior_kurtosis = 'X_train_wavelet_bior_kurtosis'
X_train_wavelet_rbio_kurtosis = 'X_train_wavelet_rbio_kurtosis'
X_train_wavelet_dmey_kurtosis = 'X_train_wavelet_dmey_kurtosis'
X_train_stats_kurtosis_kurtosis = 'X_train_stats_kurtosis_kurtosis'
X_train_wavelet_haar_std = 'X_train_wavelet_haar_std'
X_train_wavelet_db_std = 'X_train_wavelet_db_std'
X_train_wavelet_sym_std = 'X_train_wavelet_sym_std'
X_train_wavelet_coif_std = 'X_train_wavelet_coif_std'
X_train_wavelet_bior_std = 'X_train_wavelet_bior_std'
X_train_wavelet_rbio_std = 'X_train_wavelet_rbio_std'
X_train_wavelet_dmey_std = 'X_train_wavelet_dmey_std'
X_train_stats_std_std = 'X_train_stats_std_std'
X_train_stats_kurtosis = 'X_train_stats_kurtosis'
X_train_stats_min = 'X_train_stats_min'
X_train_stats_max = 'X_train_stats_max'
X_train_stats_mean = 'X_train_stats_mean'
X_train_stats_median = 'X_train_stats_median'
X_train_stats_std = 'X_train_stats_std'
X_train_stats_var = 'X_train_stats_var'
X_train_ftt_dl = 'X_train_ftt_dl'
X_train_lyarosenstein = 'X_train_lyarosenstein'
X_train_mne_stft = 'X_train_mne_stft'
X_train_LZ_primitive_complexity = 'X_train_LZ_primitive_complexity'
X_train_LZ_exhaustive_complexity = 'X_train_LZ_exhaustive_complexity'

# X_test
X_test_bow_histo = 'X_test_bow_histo'
X_test_pyeeg_dfa = 'X_test_pyeeg_dfa'
X_test_pyeeg_pfd = 'X_test_pyeeg_pfd'
X_test_pyeeg_hjorth = 'X_test_pyeeg_hjorth'
X_test_pyeeg_apen = 'X_test_pyeeg_apen'
X_test_pyeeg_sampen = 'X_test_pyeeg_sampen'
X_test_pyrem_apen = 'X_test_pyrem_apen'
X_test_pyeeg_hfd = 'X_test_pyeeg_hfd'
X_test_pyeeg_fisher_info = 'X_test_pyeeg_fisher_info'
X_test_pyeeg_svd = 'X_test_pyeeg_svd'
X_test_pyeeg_bin_power = 'X_test_pyeeg_bin_power'
X_test_pyeeg_bin_power_ratio = 'X_test_pyeeg_bin_power_ratio'
X_test_pyeeg_spect = 'X_test_pyeeg_spect'
X_test_pyeeg_hurst = 'X_test_pyeeg_hurst'
X_test_pyrem_sampen = 'X_test_pyrem_sampen'
X_test_pyrem_apen = 'X_test_pyrem_apen'
X_test_pyrem_dfa = 'X_test_pyrem_dfa'
X_test_pyrem_hjorth = 'X_test_pyrem_hjorth'
X_test_pyrem_pfd = 'X_test_pyrem_pfd'
X_test_pyrem_hurst = 'X_test_pyrem_hurst'
X_test_kolmogorov = 'X_test_kolmogorov'
X_test_wavelet_haar_rse = 'X_test_wavelet_haar_rse'
X_test_wavelet_db_rse = 'X_test_wavelet_db_rse'
X_test_wavelet_sym_rse = 'X_test_wavelet_sym_rse'
X_test_wavelet_coif_rse = 'X_test_wavelet_coif_rse'
X_test_wavelet_bior_rse = 'X_test_wavelet_bior_rse'
X_test_wavelet_rbio_rse = 'X_test_wavelet_rbio_rse'
X_test_wavelet_dmey_rse = 'X_test_wavelet_dmey_rse'
X_test_wavelet_haar_kurtosis = 'X_test_wavelet_haar_kurtosis'
X_test_wavelet_db_kurtosis = 'X_test_wavelet_db_kurtosis'
X_test_wavelet_sym_kurtosis = 'X_test_wavelet_sym_kurtosis'
X_test_wavelet_coif_kurtosis = 'X_test_wavelet_coif_kurtosis'
X_test_wavelet_bior_kurtosis = 'X_test_wavelet_bior_kurtosis'
X_test_wavelet_rbio_kurtosis = 'X_test_wavelet_rbio_kurtosis'
X_test_wavelet_dmey_kurtosis = 'X_test_wavelet_dmey_kurtosis'
X_test_stats_kurtosis_kurtosis = 'X_test_stats_kurtosis_kurtosis'
X_test_wavelet_haar_std = 'X_test_wavelet_haar_std'
X_test_wavelet_db_std = 'X_test_wavelet_db_std'
X_test_wavelet_sym_std = 'X_test_wavelet_sym_std'
X_test_wavelet_coif_std = 'X_test_wavelet_coif_std'
X_test_wavelet_bior_std = 'X_test_wavelet_bior_std'
X_test_wavelet_rbio_std = 'X_test_wavelet_rbio_std'
X_test_wavelet_dmey_std = 'X_test_wavelet_dmey_std'
X_test_stats_std_std = 'X_test_stats_std_std'
X_test_stats_kurtosis = 'X_test_stats_kurtosis'
X_test_stats_min = 'X_test_stats_min'
X_test_stats_max = 'X_test_stats_max'
X_test_stats_mean = 'X_test_stats_mean'
X_test_stats_median = 'X_test_stats_median'
X_test_stats_std = 'X_test_stats_std'
X_test_stats_var = 'X_test_stats_var'
X_test_ftt_dl = 'X_test_ftt_dl'
X_test_lyarosenstein = 'X_test_lyarosenstein'
X_test_mne_stft = 'X_test_mne_stft'
X_test_LZ_primitive_complexity = 'X_test_LZ_primitive_complexity'
X_test_LZ_exhaustive_complexity = 'X_test_LZ_exhaustive_complexity'

# X_train
X_train_features = {}
# PyEEG features
X_train_features[X_train_pyeeg_dfa]             = get_pyeeg_feature_file(app.X_train_label, pyeeg_dfa)
X_train_features[X_train_pyeeg_pfd]             = get_pyeeg_feature_file(app.X_train_label, pyeeg_pfd)
X_train_features[X_train_pyeeg_hjorth]          = get_pyeeg_feature_file(app.X_train_label, pyeeg_hjorth)
X_train_features[X_train_pyeeg_hfd]             = get_pyeeg_feature_file(app.X_train_label, pyeeg_hfd)
X_train_features[X_train_pyeeg_fisher_info]     = get_pyeeg_feature_file(app.X_train_label, pyeeg_fisher_info)
X_train_features[X_train_pyeeg_svd]             = get_pyeeg_feature_file(app.X_train_label, pyeeg_svd)
X_train_features[X_train_pyeeg_bin_power]       = get_pyeeg_feature_file(app.X_train_label, pyeeg_bin_power)
X_train_features[X_train_pyeeg_bin_power_ratio] = get_pyeeg_feature_file(app.X_train_label, pyeeg_bin_power_ratio)
X_train_features[X_train_pyeeg_spect]           = get_pyeeg_feature_file(app.X_train_label, pyeeg_spect)
X_train_features[X_train_pyeeg_hurst]           = get_pyeeg_feature_file(app.X_train_label, pyeeg_hurst)
# Wavelets
X_train_features[X_train_wavelet_haar_rse]    = get_wavelet_feature_file(app.X_train_label, wave_haar, 'RSE')
X_train_features[X_train_wavelet_db_rse]      = get_wavelet_feature_file(app.X_train_label, wave_db, 'RSE')
X_train_features[X_train_wavelet_sym_rse]     = get_wavelet_feature_file(app.X_train_label, wave_sym, 'RSE')
X_train_features[X_train_wavelet_coif_rse]    = get_wavelet_feature_file(app.X_train_label, wave_coif, 'RSE')
X_train_features[X_train_wavelet_bior_rse]    = get_wavelet_feature_file(app.X_train_label, wave_bior, 'RSE')
X_train_features[X_train_wavelet_rbio_rse]    = get_wavelet_feature_file(app.X_train_label, wave_rbio, 'RSE')
X_train_features[X_train_wavelet_dmey_rse]    = get_wavelet_feature_file(app.X_train_label, wave_dmey, 'RSE')
X_train_features[X_train_wavelet_haar_kurtosis]    = get_wavelet_feature_file(app.X_train_label, wave_haar, 'kurtosis')
X_train_features[X_train_wavelet_db_kurtosis]      = get_wavelet_feature_file(app.X_train_label, wave_db, 'kurtosis')
X_train_features[X_train_wavelet_sym_kurtosis]     = get_wavelet_feature_file(app.X_train_label, wave_sym, 'kurtosis')
X_train_features[X_train_wavelet_coif_kurtosis]    = get_wavelet_feature_file(app.X_train_label, wave_coif, 'kurtosis')
X_train_features[X_train_wavelet_bior_kurtosis]    = get_wavelet_feature_file(app.X_train_label, wave_bior, 'kurtosis')
X_train_features[X_train_wavelet_rbio_kurtosis]    = get_wavelet_feature_file(app.X_train_label, wave_rbio, 'kurtosis')
X_train_features[X_train_wavelet_dmey_kurtosis]    = get_wavelet_feature_file(app.X_train_label, wave_dmey, 'kurtosis')
X_train_features[X_train_wavelet_haar_std]    = get_wavelet_feature_file(app.X_train_label, wave_haar, 'std')
X_train_features[X_train_wavelet_db_std]      = get_wavelet_feature_file(app.X_train_label, wave_db, 'std')
X_train_features[X_train_wavelet_sym_std]     = get_wavelet_feature_file(app.X_train_label, wave_sym, 'std')
X_train_features[X_train_wavelet_coif_std]    = get_wavelet_feature_file(app.X_train_label, wave_coif, 'std')
X_train_features[X_train_wavelet_bior_std]    = get_wavelet_feature_file(app.X_train_label, wave_bior, 'std')
X_train_features[X_train_wavelet_rbio_std]    = get_wavelet_feature_file(app.X_train_label, wave_rbio, 'std')
X_train_features[X_train_wavelet_dmey_std]    = get_wavelet_feature_file(app.X_train_label, wave_dmey, 'std')
# Pyrem features 
X_train_features[X_train_pyrem_apen] = np.array(np.loadtxt(app.X_train_label + '_pyrem_apen.csv', delimiter=";"))
X_train_features[X_train_pyrem_sampen] = np.array(np.loadtxt(X_label_train + '_pyrem_sampen.csv', delimiter=";"))
X_train_features[X_train_pyrem_dfa] = np.array(np.loadtxt(app.X_train_label + '_pyrem_dfa.csv', delimiter=";"))
X_train_features[X_train_pyrem_hjorth] = np.array(np.loadtxt(app.X_train_label + '_pyrem_hjorth.csv', delimiter=";"))
X_train_features[X_train_pyrem_pfd] = np.array(np.loadtxt(app.X_train_label + '_pyrem_pfd.csv', delimiter=";")) 
X_train_features[X_train_pyrem_hurst] = np.array(np.loadtxt(app.X_train_label + '_pyrem_hurst.csv', delimiter=";"))
# other 
X_train_features[X_train_kolmogorov] = np.array(np.loadtxt(app.X_train_label + '_kolmogorov.csv', delimiter=";"))
X_train_features[X_train_ftt_dl] = np.array(np.loadtxt(app.X_train_label + '_fft_dl.csv', delimiter=";"))
X_train_features[X_train_lyarosenstein] = np.array(np.loadtxt(app.X_train_label + '_lyarosenstein.csv', delimiter=";"))
X_train_features[X_train_mne_stft] = np.array(np.loadtxt(app.X_train_label + '_mne_stft.csv', delimiter=";"))
X_train_features[X_train_LZ_primitive_complexity] = np.array(np.loadtxt(app.X_train_label + '_LZ_primitive_complexity.csv', delimiter=","))
X_train_features[X_train_LZ_exhaustive_complexity] = np.array(np.loadtxt(app.X_train_label + '_LZ_exhaustive_complexity.csv', delimiter=","))
# statistical features
X_train_features[X_train_stats_kurtosis]   = np.array(stats.kurtosis(X_train, axis=1))
X_train_features[X_train_stats_min]        = np.array(np.min(X_train, axis=1))
X_train_features[X_train_stats_max]        = np.array(np.max(X_train, axis=1))
X_train_features[X_train_stats_mean]       = np.array(np.mean(X_train, axis=1))
X_train_features[X_train_stats_median]     = np.array(np.median(X_train, axis=1))
X_train_features[X_train_stats_std]        = np.array(np.std(X_train, axis=1))
X_train_features[X_train_stats_var]        = np.array(np.var(X_train, axis=1))

XX_train = []
for features in X_train_features:
    XX_train=np.c_[X_train_features[features]]
XX_train = np.array(XX_train)

# X_test
X_test_features = {}
# PyEEG features
X_test_features[X_test_pyeeg_dfa]             = get_pyeeg_feature_file(app.X_test_label, pyeeg_dfa)
X_test_features[X_test_pyeeg_pfd]             = get_pyeeg_feature_file(app.X_test_label, pyeeg_pfd)
X_test_features[X_test_pyeeg_hjorth]          = get_pyeeg_feature_file(app.X_test_label, pyeeg_hjorth)
X_test_features[X_test_pyeeg_hfd]             = get_pyeeg_feature_file(app.X_test_label, pyeeg_hfd)
X_test_features[X_test_pyeeg_fisher_info]     = get_pyeeg_feature_file(app.X_test_label, pyeeg_fisher_info)
X_test_features[X_test_pyeeg_svd]             = get_pyeeg_feature_file(app.X_test_label, pyeeg_svd)
X_test_features[X_test_pyeeg_bin_power]       = get_pyeeg_feature_file(app.X_test_label, pyeeg_bin_power)
X_test_features[X_test_pyeeg_bin_power_ratio] = get_pyeeg_feature_file(app.X_test_label, pyeeg_bin_power_ratio)
X_test_features[X_test_pyeeg_spect]           = get_pyeeg_feature_file(app.X_test_label, pyeeg_spect)
X_test_features[X_test_pyeeg_hurst]           = get_pyeeg_feature_file(app.X_test_label, pyeeg_hurst)
# Wavelets
X_test_features[X_test_wavelet_haar_rse]    = get_wavelet_feature_file(app.X_test_label, wave_haar, 'RSE')
X_test_features[X_test_wavelet_db_rse]      = get_wavelet_feature_file(app.X_test_label, wave_db, 'RSE')
X_test_features[X_test_wavelet_sym_rse]     = get_wavelet_feature_file(app.X_test_label, wave_sym, 'RSE')
X_test_features[X_test_wavelet_coif_rse]    = get_wavelet_feature_file(app.X_test_label, wave_coif, 'RSE')
X_test_features[X_test_wavelet_bior_rse]    = get_wavelet_feature_file(app.X_test_label, wave_bior, 'RSE')
X_test_features[X_test_wavelet_rbio_rse]    = get_wavelet_feature_file(app.X_test_label, wave_rbio, 'RSE')
X_test_features[X_test_wavelet_dmey_rse]    = get_wavelet_feature_file(app.X_test_label, wave_dmey, 'RSE')
X_test_features[X_test_wavelet_haar_kurtosis]    = get_wavelet_feature_file(app.X_test_label, wave_haar, 'kurtosis')
X_test_features[X_test_wavelet_db_kurtosis]      = get_wavelet_feature_file(app.X_test_label, wave_db, 'kurtosis')
X_test_features[X_test_wavelet_sym_kurtosis]     = get_wavelet_feature_file(app.X_test_label, wave_sym, 'kurtosis')
X_test_features[X_test_wavelet_coif_kurtosis]    = get_wavelet_feature_file(app.X_test_label, wave_coif, 'kurtosis')
X_test_features[X_test_wavelet_bior_kurtosis]    = get_wavelet_feature_file(app.X_test_label, wave_bior, 'kurtosis')
X_test_features[X_test_wavelet_rbio_kurtosis]    = get_wavelet_feature_file(app.X_test_label, wave_rbio, 'kurtosis')
X_test_features[X_test_wavelet_dmey_kurtosis]    = get_wavelet_feature_file(app.X_test_label, wave_dmey, 'kurtosis')
X_test_features[X_test_wavelet_haar_std]    = get_wavelet_feature_file(app.X_test_label, wave_haar, 'std')
X_test_features[X_test_wavelet_db_std]      = get_wavelet_feature_file(app.X_test_label, wave_db, 'std')
X_test_features[X_test_wavelet_sym_std]     = get_wavelet_feature_file(app.X_test_label, wave_sym, 'std')
X_test_features[X_test_wavelet_coif_std]    = get_wavelet_feature_file(app.X_test_label, wave_coif, 'std')
X_test_features[X_test_wavelet_bior_std]    = get_wavelet_feature_file(app.X_test_label, wave_bior, 'std')
X_test_features[X_test_wavelet_rbio_std]    = get_wavelet_feature_file(app.X_test_label, wave_rbio, 'std')
X_test_features[X_test_wavelet_dmey_std]    = get_wavelet_feature_file(app.X_test_label, wave_dmey, 'std')
# Pyrem features 
X_test_features[X_test_pyrem_apen] = np.array(np.loadtxt(app.X_test_label + '_pyrem_apen.csv', delimiter=";"))
X_test_features[X_test_pyrem_sampen] = np.array(np.loadtxt(app.X_test_label + '_pyrem_sampen.csv', delimiter=";"))
X_test_features[X_test_pyrem_dfa] = np.array(np.loadtxt(app.X_test_label + '_pyrem_dfa.csv', delimiter=";"))
X_test_features[X_test_pyrem_hjorth] = np.array(np.loadtxt(app.X_test_label + '_pyrem_hjorth.csv', delimiter=";"))
X_test_features[X_test_pyrem_pfd] = np.array(np.loadtxt(app.X_test_label + '_pyrem_pfd.csv', delimiter=";")) 
X_test_features[X_test_pyrem_hurst] = np.array(np.loadtxt(app.X_test_label + '_pyrem_hurst.csv', delimiter=";"))
# other
X_test_features[X_test_kolmogorov] = np.array(np.loadtxt(app.X_test_label + '_kolmogorov.csv', delimiter=";"))
X_test_features[X_test_ftt_dl] = np.array(np.loadtxt(app.X_test_label + '_fft_dl.csv', delimiter=";"))
X_test_features[X_test_lyarosenstein] = np.array(np.loadtxt(app.X_test_label + '_lyarosenstein.csv', delimiter=";"))
X_test_features[X_test_mne_stft] = np.array(np.loadtxt(app.X_test_label + '_mne_stft.csv', delimiter=";"))
X_test_features[X_test_LZ_primitive_complexity] = np.array(np.loadtxt(app.X_test_label + '_LZ_primitive_complexity.csv', delimiter=","))
X_test_features[X_test_LZ_exhaustive_complexity] = np.array(np.loadtxt(app.X_test_label + '_LZ_exhaustive_complexity.csv', delimiter=","))
# statistical features
X_test_features[X_test_stats_kurtosis]   = np.array(stats.kurtosis(X_test, axis=1))
X_test_features[X_test_stats_min]        = np.array(np.min(X_test, axis=1))
X_test_features[X_test_stats_max]        = np.array(np.max(X_test, axis=1))
X_test_features[X_test_stats_mean]       = np.array(np.mean(X_test, axis=1))
X_test_features[X_test_stats_median]     = np.array(np.median(X_test, axis=1))
X_test_features[X_test_stats_std]        = np.array(np.std(X_test, axis=1))
X_test_features[X_test_stats_var]        = np.array(np.var(X_test, axis=1))

XX_test = []
for features in X_test_features:
    XX_test=np.c_[X_test_features[features]]
XX_test = np.array(XX_test)

print XX_train.shape, XX_test.shape


##########################################################################################
# FEATURES SELECTION
##########################################################################################

def test_classifier_CV(XX_train, y_train, clf, kfold=3, normalize=False):
    if normalize:
        XX_train_selected = StandardScaler(with_mean=True, with_std=True).fit_transform(XX_train)
    else:
        XX_train_selected = XX_train
    return np.mean(np.array(cross_validation.cross_val_score(clf, XX_train_selected, y_train, cv=kfold)))
    
def test_feature(XX_train_features):
    all_res = []
    all_res.append(test_classifier_CV(XX_train, y_train, linear_model.LogisticRegression()))
    all_res.append(test_classifier_CV(XX_train, y_train, linear_model.LogisticRegression(penalty='l1', C=11)))
    all_res.append(test_classifier_CV(XX_train, y_train, GradientBoostingClassifier()))
    all_res.append(test_classifier_CV(XX_train, y_train, RandomForestClassifier()))
    all_res.append(test_classifier_CV(XX_train_features, y_train, svm.SVC(), normalize=True))
    res_mean = np.mean(all_res)
    print 'linear_model               : ', str(all_res[0])
    print 'linear_model               : ', str(all_res[1])
    print 'GradientBoostingClassifier : ', str(all_res[2])
    print 'RandomForestClassifier     : ', str(all_res[3])
    print 'SVC                        : ', str(all_res[4])
    print 'Average                    : ', str(res_mean)
    return res_mean
    
def test_each_feature_alone():
    n = 1
    features_to_keep = []
    for feature in X_train_features:
        X_train_features_2 = []
        X_train_features_2 = np.c_[X_train_features[feature]]
        test_feature(X_train_features_2)
        n+=1
    print 'important features : ', features_to_keep, ' (', len(features_to_keep), '/', str(len(X_train_features)), ')'
    
def greedy_select_features(X_train_features, X_train_features_greedy, score):
    print 'greedy_select_features : X_train_features(', len(X_train_features), ') X_train_features_greedy(', len(X_train_features_greedy), ') score:', str(score)
    feature_to_keep = None
    for feature in X_train_features:
        X_train_features_temp = copy.copy(X_train_features_greedy)
        X_train_features_temp = np.c_[X_train_features_temp, X_train_features[feature]]
        score_temp = test_feature(X_train_features_temp)
        if score_temp > score:
            score = score_temp
            feature_to_keep = feature
    if feature_to_keep:
        print 'feature_to_keep : ', feature_to_keep
        X_train_features_greedy = np.c_[X_train_features_greedy, X_train_features[feature_to_keep]]
        X_train_features.pop(feature_to_keep)
        greedy_select_features(X_train_features_greedy, X_train_features, score)
    return X_train_features_greedy
        
def launch_greedy():
    X_train_features_greedy = []
    X_train_features_2 = copy.copy(X_train_features)
    X_train_features_greedy_final = greedy_select_features(X_train_features_2, X_train_features_greedy, 0.0)
    np.savetxt('X_train_greedy.csv', np.array(X_train_features_greedy_final.items), fmt='%s')


##########################################################################################
# CLASSIFIER
##########################################################################################

# normalize
XX_train_normalized = (XX_train - XX_train.mean(axis=0)) / XX_train.std(axis=0)
XX_test_normalized = (XX_test - XX_test.mean(axis=0)) / XX_test.std(axis=0)

def mean_score(predictions, label_int):
    cpt = {}
    for label in label_int:
        cpt[label] = 0
    for pred in predictions:
        cpt[pred] += 1
    return max(cpt.iteritems(), key=operator.itemgetter(1))[0]
    
def vote_classifiers():
    label_int = {'W':0, 'R':1, 'N1':2, 'N2':3, 'N3':4}
    y_pred_log = np.array(np.genfromtxt('y_pred_log.txt', dtype='str'))
    y_pred_svc = np.array(np.genfromtxt('y_pred_svc.txt', dtype='str'))
    y_pred_rf = np.array(np.genfromtxt('y_pred_rf.txt', dtype='str'))
    y_pred_gb = np.array(np.genfromtxt('y_pred_gb.txt', dtype='str'))
    y_pred_bow = np.array(np.genfromtxt('y_pred_bow.txt', dtype='str'))
    y_pred_final = []
    for i in range(0, len(y_pred_log)):
        res_log = y_pred_log[i]
        res_svc = y_pred_svc[i]
        res_rf = y_pred_rf[i]
        res_gb = y_pred_gb[i]
        res_bow = y_pred_bow[i]
        predictions = [res_log, res_svc, res_rf, res_gb, res_bow]
        if 'N1' in predictions:
            y_pred_final.append('N1')
        else:
            y_pred_final.append(mean_score(predictions, label_int))
    y_pred_final = np.array(y_pred_final)
    np.savetxt('y_pred_final.txt', y_pred_final, fmt='%s')
    print y_pred_final[:100] 
        
# test somes classifier with CV
print 'linear_model               : ', test_classifier_CV(linear_model.LogisticRegression(penalty='l1', C=11))
print 'GradientBoostingClassifier : ', test_classifier_CV(GradientBoostingClassifier())
print 'RandomForestClassifier     : ', test_classifier_CV(RandomForestClassifier(n_estimators=45))
print 'SVC                        : ', test_classifier_CV(svm.SVC(), normalize=True)
print 'linear_model               : ', test_classifier_CV(linear_model.LogisticRegression(penalty='l2', C=11))

# apply final prediction    
clf = SVC(C=10, gamma=0.0)
XX_train_normalized = (XX_train - XX_train.mean(axis=0)) / XX_train.std(axis=0)
XX_test_normalized = (XX_test - XX_test.mean(axis=0)) / XX_test.std(axis=0)
y_pred = clf.predict(XX_test)

# save y_pred
np.savetxt('y_pred_svc.txt', y_pred, fmt='%s')

print y_pred_final[:100] 

