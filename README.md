# eeg-machine-learning-challenge
EEG Machine Learning Challenge, Python and MATLAB (Scholar project, 2015)

**The goal of this challenge consists to determine the right sleep stage (W, R, N1, N2, N3) from EEG signal of 30 seconds; dataset contains 10178 and 10087 EEG for train and test respectively.**

### Features extraction
First of all, I have written my own Python script (see: cluster util.py, https://github.com/martin-prillard/shavadoop) to apply distributed and parallel computing on the school's network, like a cluster. It divide the computing time by more than 100
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
