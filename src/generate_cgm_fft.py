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
from scipy.spatial.distance import cdist, euclidean
from fastdtw import fastdtw
from dtaidistance import dtw

import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import MDS

from tqdm import tqdm

import re
import pandas as pd


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
os.makedirs('./fig/cgm/fft', exist_ok=True)

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


def dtw_distance_from_traces(x, y, fs, n_fft=None, window="hann", use_c=True):
    """
    Fast DTW distance between two equal-frequency traces.

    Parameters
    ----------
    x, y : array-like
        1D traces sampled at the same frequency.
    fs : float
        Sampling frequency (kept for API compatibility; not used here).
    n_fft : int or None
        If provided, compare magnitude spectra instead of raw traces.
    window : str
        Window name for FFT preprocessing.
    use_c : bool
        Use the faster C-backed DTW implementation.

    Returns
    -------
    float
        DTW distance
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if n_fft is not None:
        win_x = np.hanning(len(x)) if window == "hann" else np.ones(len(x))
        win_y = np.hanning(len(y)) if window == "hann" else np.ones(len(y))
        x = np.abs(np.fft.rfft(x * win_x, n=n_fft))
        y = np.abs(np.fft.rfft(y * win_y, n=n_fft))
    return dtw.distance_fast(x, y, use_c=use_c)



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


# Now compute DTW distance from FFT for all traces
Dx = np.zeros(shape=(len(all_traces),len(all_traces)))

for i in tqdm(range(len(all_traces))):
    for j in range(len(all_traces)):
        Dx[i,j] = dtw_distance_from_traces(all_traces[i], all_traces[j], SAMPLING_FREQUENCY_HZ)


# Plot MDS from distances


traces = [np.asarray(t, dtype=np.float64) for t in all_traces]

# Full NxN DTW distance matrix using all available cores
Dx = dtw.distance_matrix_fast(
    traces,
    compact=False,
    parallel=True,
    use_c=True,
)

np.savetxt('./calc/cgm/fft/dtw_distance.csv', Dx, delimiter=',', fmt='%.4f')

# Set inf values to the maximum distance value manually
max_finite = np.max(Dx[np.isfinite(Dx)])
Dx[~np.isfinite(Dx)] = max_finite


# D = your (n x n) distance matrix
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)

X = mds.fit_transform(Dx)

plt.scatter(X[:, 0], X[:, 1])
plt.title("MDS embedding")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

################################################################################
## COMPARE WITH DIABETES PARAMETERS
################################################################################

# Extract subject information from the PATH
pattern = re.compile(r"dexcom_g6/([0-9]+)/")
subjects = [pattern.search(s).groups()[0] for s in paths if pattern.search(s)]


clinical_df = pd.read_csv('./data/aireadi/clinical_data/measurement.csv')

# Creatinine: 
# "measurement_source_value" = "Urine Creatinine (mg/dL)"
creatinine_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "Urine Creatinine (mg/dL)"]
for i in range(tmp_df.shape[0]):
    creatinine_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]

# HbA1c (%)
# "measurement_source_value" = "HbA1c (%)"
hgba1c_map = {}
tmp_df = clinical_df[clinical_df["measurement_source_value"] == "HbA1c (%)"]
for i in range(tmp_df.shape[0]):
    hgba1c_map[str(tmp_df['person_id'].iloc[i])] = tmp_df['value_as_number'].iloc[i]


plot_df = pd.DataFrame.from_dict({
    "Subject": subjects,
    "MDS1": X[:, 0],
    "MDS2": X[:, 1],
    "UrineCr": [creatinine_map[s] if s in creatinine_map else np.nan for s in subjects],
    "HgbA1c": [hgba1c_map[s] if s in hgba1c_map else np.nan for s in subjects],
})

plot_df.to_csv('./calc/cgm/fft/mds_plot_df.csv')

sc = plt.scatter(
    plot_df['MDS1'],
    plot_df['MDS2'],
    c=plot_df['HgbA1c'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("HgbA1c")

plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS embedding colored by HgbA1c")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/fft/hgba1c_mds_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/fft/hgba1c_mds_plot.svg")            # vector SVG

plt.show()

sc = plt.scatter(
    plot_df['MDS1'],
    plot_df['MDS2'],
    c=plot_df['UrineCr'],
    cmap="viridis",   # color map
)

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label("Urine [Cr]")

plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.title("MDS embedding colored by Urine [Cr]")

plt.tight_layout()

# Save in multiple formats
plt.savefig("./fig/cgm/fft/creatinine_mds_plot.png", dpi=300)   # high-res PNG
plt.savefig("./fig/cgm/fft/creatinine_mds_plot.svg")            # vector SVG

plt.show()


print("All done!")