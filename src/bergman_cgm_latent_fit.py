#!/usr/bin/env pyhton
#
# Fit mechanistic ODE model to CGM data to extract biologically meaningful
# parameters upon which to perform later analysis. Mechanistic model is 
# based on a Bergman minimal model, gold standard mechanistic ODE for
# glucose monitoring.
#
# `ReducedBergmanFit` class instances contain the fitted parameters
#
# Methods included to generate traces for glucose curves, and key latent
# parameters of meal drive m(t) and latent insulin x(t).
#
# @author Rahul Dhodapkar
#

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path

from tqdm import tqdm

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

import copy

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
## DEFINE CONSTANTS
################################################################################

VALUE_HIGH = 500
VALUE_LOW = 20
SAMPLING_FREQUENCY_HZ = 1/(5 * 60) # q5mins
EXPECTED_NUMBER_OF_FREQS_OUTPUT = 357

torch.set_default_dtype(torch.float64)
np.random.seed(7)
torch.manual_seed(7)

################################################################################
## BUILD OUTPUT SCAFFOLDING
################################################################################

os.makedirs('./calc/cgm/bergman_fit', exist_ok=True)
os.makedirs('./fig/cgm/bergman_fit', exist_ok=True)

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


################################################################################
## FIT BERGMAN MODEL
################################################################################

p1_vals = []
p2_vals = []
p3_vals = []

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32


def tv(x):
    return torch.sum(torch.abs(x[1:] - x[:-1]))

def l2diff(x):
    return torch.sum((x[1:] - x[:-1]) ** 2)

class Model(nn.Module):
    def __init__(self, T, dt):
        super().__init__()
        self.T = T
        self.dt = float(dt)
        #
        self.log_p1 = nn.Parameter(torch.tensor(-2.5, dtype=dtype))
        self.log_p2 = nn.Parameter(torch.tensor(-2.0, dtype=dtype))
        self.log_p3 = nn.Parameter(torch.tensor(-1.5, dtype=dtype))
        self.meal_logits = nn.Parameter(torch.full((T,), -5.0, dtype=dtype))
        self.insulin_logits = nn.Parameter(torch.full((T,), -4.5, dtype=dtype))
        #
    def forward(self):
        p1 = F.softplus(self.log_p1) + 1e-4
        p2 = F.softplus(self.log_p2) + 1e-4
        p3 = F.softplus(self.log_p3) + 1e-4
        #
        Ra = 0.08 * F.softplus(self.meal_logits)
        I = 0.08 * F.softplus(self.insulin_logits)
        #
        dev = self.log_p1.device
        g_list = [torch.zeros((), device=dev, dtype=dtype)]
        X_list = [torch.zeros((), device=dev, dtype=dtype)]
        #
        for k in range(self.T - 1):
            g_next = g_list[-1] + self.dt * (-(p1 + X_list[-1]) * g_list[-1] + Ra[k])
            X_next = X_list[-1] + self.dt * (-p2 * X_list[-1] + p3 * I[k])
            g_list.append(g_next)
            X_list.append(X_next)
        #
        g_hat = torch.stack(g_list)
        X_hat = torch.stack(X_list)
        y_hat = g_hat
        return y_hat, g_hat, X_hat, Ra, I, (p1, p2, p3)

# all_traces = all_traces[:100]

for i in tqdm(range(len(all_traces))):
    T = min(len(all_traces[i]), 288) #clip to 24hrs if recording longer than that
    dt = 1.0
    hours = np.arange(T) * 5.0 / 60.0 # sampling every 5 minutes
    glucose = np.array(all_traces[i][:T])
    #
    # normalize for fitting stability
    mu = glucose.mean()
    sd = glucose.std() + 1e-6
    y = torch.tensor((glucose - mu) / sd, dtype=dtype)
    #
    y = y.to(device=device, dtype=dtype)
    #
    model = Model(T, dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    #
    for step in range(500):
        opt.zero_grad()
        #
        y_hat, g_hat, X_hat, Ra, I, params = model()
        mse = torch.mean((y_hat - y) ** 2)
        #
        loss = (
            mse
            + 0.10 * torch.mean(Ra)
            + 0.03 * tv(Ra) / T
            + 0.02 * l2diff(I) / T
            + 0.01 * torch.mean(I)
            + 0.002 * l2diff(X_hat) / T
        )
        #
        loss.backward()
        opt.step()
        #
        if step % 100 == 0 or step == 499:
            p1, p2, p3 = [v.item() for v in params]
            print(step, loss.item(), mse.item(), p1, p2, p3)
    #
    with torch.no_grad():
        y_hat, g_hat, X_hat, Ra, I, params = model()
    #
    p1, p2, p3 = params
    p1_vals.append(p1.to('cpu').item())
    p2_vals.append(p2.to('cpu').item())
    p3_vals.append(p3.to('cpu').item())
    #
    # rescale back to mg/dL
    pred = y_hat.to('cpu').numpy() * sd + mu
    meal = Ra.to('cpu').numpy()
    ins = I.to('cpu').numpy()
    Xhat = X_hat.to('cpu').numpy()
    #
    outdir = Path('./fig/cgm/bergman_fit')
    fig1 = outdir / 'fit_{}.png'.format(i)
    fig2 = outdir / 'latent_{}.png'.format(i)
    #
    plt.figure(figsize=(11, 4))
    plt.plot(hours, glucose, label='Observed CGM')
    plt.plot(hours, pred, label='Fitted glucose')
    plt.xlabel('Time (hours)')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    #
    plt.figure(figsize=(11, 4.5))
    plt.plot(hours, meal, label='Latent meal appearance')
    plt.plot(hours, ins, label='Latent insulin drive')
    plt.plot(hours, Xhat, label='Remote insulin action')
    plt.xlabel('Time (hours)')
    plt.ylabel('Latent units')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()
    #
    torch.save(model, './calc/cgm/bergman_fit/model_{}.pth'.format(i))

df = pd.DataFrame({
    'p1': p1_vals,
    'p2': p2_vals,
    'p3': p3_vals
})
df.to_csv('./calc/cgm/bergman_fit/all_fit_dataframe.csv', index=False)

print('All done!')
