# eeg-machine-learning-challenge
EEG Machine Learning Challenge, Python and MATLAB (Scholar project, 2015)

**The goal of this challenge consisted to determine the right sleep stage (W, R, N1, N2, N3) from EEG signal of 30 seconds; dataset contains 10178 and 10087 EEG for train and test respectively.**

### Features extraction
First of all, I have written my own Python script (see: https://github.com/martin-prillard/cluster-computing) to apply distributed and parallel computing on the school's network, like a cluster. It divide the computing time by more than 100
than 100.

#### PyEEG / PyREM / MNE
Thanks to the PyEEG library, I extracted these features:
- DFA: Detrended Fuctuation Analysis
- PFD: Petrosian Fractal Dimension
- hjorth: Hjorth mobility and complexity
- ApEn: Approximate entropy
- SampEn: sample entropy (SampEn)
- hfd: Higuchi Fractal Dimension
- Fsher info: Fisher information
- SVD: SVD Entropy
- bin power & bin power ratio: power in each frequency bin specifed by Band from FFT result
- PSD: Power Spectral Density, spectrum power in a set of frequency bins
- spect: spectral entropy of a time series
- hurst: Hurst exponent of X

Then, I replaced some of PyEEG's features by PyREM's features, more reliable. Thanks to my cluster computing script, it was very fast (even for entropy calculation).
Sparse Fast Fourier Transform from MNE and the wavelets (RSE/kurtosis/std) were also very useful.

### BOW-Kmeans
Local patches extracted from images or videos are treated as words and the codebook is constructed by clustering all the local patches in the training dataset. Similar to the extension of the bag of words representation in computer vision, we here extend the bag-of-words representation to characterize biomedical time series by regarding local segments extracted from time series as words and treat the time series as documents. Codebook is a set of predefined words, which are also called codewords, compute by k-means.

Consequently, I implemented all the work of Jin Wang in Python (available for Matlab), and distributed it on my cluster. I distributed local segments extraction, which computes wavelet coefficient, then the Fast Fourier Transform (FFT) from
each segments.
The codebook was build by clustering (mini-batch Kmeans++, 1500 centroids and 24 millions of segments) local segments from training data. The next step, also distributed, assigned one codeword for each local segment, to finally build one histogram
representation for each time series (sum the number of each codeword that exists in the time series). Histogram's size depends of the codebook size (200-1500).

#### Other features attempts
I used Matlab algorithms to compute the Rosensteins Lyapunov exponents, the Lempel-Ziv (measure of binary sequence complexity) and wiener Kolmogorov, which are supposed to be truly relevant.
I also tried the NMF and Beta NMF with Itakura-Saito divergente. Finally, I tried MLLab and spindles.

### Features selection
I had 317 features, without the BOW histograms features. Consequently, I did a feature selection to decrease model's complexity and avoid overfitting. Firstly, I tried a PCA, SelectKBest, RFECV (Feature ranking with recursive feature elimination and cross-validated selection of the best number of features), and ExtraTreesClassifer to keep only the 10-100th discriminatory features. Finally, Iimplemented my own Greedy Algorithm to increase the Cross-validation score on the train sample.

### Classification
I kept for a long time with a simple logistic regression, it gave me my best score (more than 0.77).
Finally, I used a C-Support Vector Classifcation (the multiclass support is handled according to a one-vs-one scheme). I used GridSearchCV to determine which hyper-parameter were the best for each models. It has to be noted that the GridSearchCV included Cross validation scores. According to my matrix confusion, the real challenge was to find the N1 sleep stage.
