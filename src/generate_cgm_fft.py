#!/usr/bin/env python
#
# Read continuous glucose monitoring data and perform FFT
# for use in spectral clustering and analysis
#
# @author Rahul Dhodapkar
#

import os
import json

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from tqdm import tqdm

################################################################################
## DEFINE CONSTANTS
################################################################################

VALUE_HIGH = 500
VALUE_LOW = 20
SAMPLING_FREQUENCY_HZ = 1/(5 * 60) # q5mins
EXPECTED_NUMBER_OF_FREQS_OUTPUT = 357

################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/cgm/fft', exist_ok=True)

################################################################################
## HELPER FUNCTIONS
################################################################################

def get_file_paths_os_walk(directory):
    """ Takes a base directory and returns all image paths. """
    file_paths = []
    file_extensions = ('.json') 
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(file_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths


def hellinger_distance_from_traces(x, y, fs, n_fft=None, window="hann"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    #
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    #
    # Remove mean
    x = x - np.mean(x)
    y = y - np.mean(y)
    #
    # Choose common FFT length
    if n_fft is None:
        n_fft = min(len(x), len(y))
    #
    # Windowing (applied before padding)
    if window == "hann":
        x = x * np.hanning(len(x))
        y = y * np.hanning(len(y))
    elif window != "none":
        raise ValueError("window must be 'hann' or 'none'")
    #
    # FFT (zero-padding handled automatically by n argument)
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    #
    # Power spectra
    p = np.abs(X) ** 2
    q = np.abs(Y) ** 2
    #
    # Normalize to probability distributions
    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)
    #
    # Hellinger distance
    h = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0)
    #
    return float(h)


################################################################################
## CALCULATE DISTANCES
################################################################################

paths = get_file_paths_os_walk('./data/aireadi/wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6')

all_traces = []

for path in tqdm(paths):
    with open(path, 'r') as file:
        # Use json.load() to convert the file content to a Python dictionary
        data = json.load(file)
        trace = []
        for i in range(len(data['body']['cgm'])):
            tmp_value = data['body']['cgm'][i]['blood_glucose']['value']
            if tmp_value == 'High':
                tmp_value = VALUE_HIGH
            elif tmp_value == 'Low':
                tmp_value = VALUE_LOW
            trace.append(tmp_value)
        all_traces.append(trace)


# Now compute hellinger distance from FFT for all traces
Dx = np.zeros(shape=(len(all_traces),len(all_traces)))

for i in tqdm(range(len(all_traces))):
    for j in range(len(all_traces)):
        Dx[i,j] = hellinger_distance_from_traces(all_traces[i], all_traces[j], SAMPLING_FREQUENCY_HZ)


# Plot MDS from distances



# D = your (n x n) distance matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)

X = mds.fit_transform(Dx)

plt.scatter(X[:, 0], X[:, 1])
plt.title("MDS embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


