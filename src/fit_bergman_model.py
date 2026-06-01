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

torch.set_default_dtype(torch.float64)
np.random.seed(7)
torch.manual_seed(7)

T = 180  # 15 hours at 5-min sampling
dt = 1.0
hours = np.arange(T) * 5.0 / 60.0

# demo ground truth
meal_events = np.zeros(T)
for idx, amp in [(18, 2.0), (58, 1.4), (110, 2.2), (150, 1.5)]:
    meal_events[idx] += amp

def gamma_kernel(length=50, a=3.5, b=5.5):
    x = np.arange(length)
    k = (x ** (a - 1)) * np.exp(-x / b)
    k[0] = 0.0
    k /= k.sum()
    return k

meal_kernel = gamma_kernel()
true_Ra = np.convolve(meal_events, meal_kernel, mode='full')[:T]
true_I = np.zeros(T)
for idx, amp in [(18, 0.8), (58, 0.6), (110, 0.9), (150, 0.5)]:
    w = np.arange(T - idx)
    true_I[idx:] += amp * np.exp(-w / 16.0)

Gb_true = 104.0
p1_true, p2_true, p3_true = 0.04, 0.12, 0.22

g = np.zeros(T)
X = np.zeros(T)
for k in range(T - 1):
    g[k + 1] = g[k] + dt * (-(p1_true + X[k]) * g[k] + 0.12 * true_Ra[k])
    X[k + 1] = X[k] + dt * (-p2_true * X[k] + p3_true * 0.12 * true_I[k])

noise = np.random.normal(0, 1.8, size=T)
glucose = Gb_true + g + noise

# normalize for fitting stability
mu = glucose.mean()
sd = glucose.std() + 1e-6
y = torch.tensor((glucose - mu) / sd)

class Model(nn.Module):
    def __init__(self, T, dt):
        super().__init__()
        self.T = T
        self.dt = dt
        self.log_p1 = nn.Parameter(torch.tensor(-2.5))
        self.log_p2 = nn.Parameter(torch.tensor(-2.0))
        self.log_p3 = nn.Parameter(torch.tensor(-1.5))
        self.meal_logits = nn.Parameter(torch.full((T,), -5.0))
        self.insulin_logits = nn.Parameter(torch.full((T,), -4.5))

    def forward(self):
        p1 = torch.nn.functional.softplus(self.log_p1) + 1e-4
        p2 = torch.nn.functional.softplus(self.log_p2) + 1e-4
        p3 = torch.nn.functional.softplus(self.log_p3) + 1e-4
        Ra = 0.08 * torch.nn.functional.softplus(self.meal_logits)
        I = 0.08 * torch.nn.functional.softplus(self.insulin_logits)

        g_list = [torch.zeros((), dtype=torch.float64)]
        X_list = [torch.zeros((), dtype=torch.float64)]
        for k in range(self.T - 1):
            g_next = g_list[-1] + self.dt * (-(p1 + X_list[-1]) * g_list[-1] + Ra[k])
            X_next = X_list[-1] + self.dt * (-p2 * X_list[-1] + p3 * I[k])
            g_list.append(g_next)
            X_list.append(X_next)
        g_hat = torch.stack(g_list)
        X_hat = torch.stack(X_list)
        y_hat = g_hat
        return y_hat, g_hat, X_hat, Ra, I, (p1, p2, p3)


def tv(x):
    return torch.sum(torch.abs(x[1:] - x[:-1]))

def l2diff(x):
    return torch.sum((x[1:] - x[:-1]) ** 2)

model = Model(T, dt)
opt = torch.optim.Adam(model.parameters(), lr=0.02)

for step in range(500):
    opt.zero_grad()
    y_hat, g_hat, X_hat, Ra, I, params = model()
    mse = torch.mean((y_hat - y) ** 2)
    # encourage sparse meals and smooth insulin
    loss = mse + 0.10 * torch.mean(Ra) + 0.03 * tv(Ra) / T + 0.02 * l2diff(I) / T + 0.01 * torch.mean(I) + 0.002 * l2diff(X_hat) / T
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 499:
        p1, p2, p3 = [v.item() for v in params]
        print(step, float(loss), float(mse), p1, p2, p3)

with torch.no_grad():
    y_hat, g_hat, X_hat, Ra, I, params = model()

# rescale back to mg/dL
pred = y_hat.numpy() * sd + mu
meal = Ra.numpy()
ins = I.numpy()
Xhat = X_hat.numpy()

outdir = Path('/tmp')
fig1 = outdir / 'fit.png'
fig2 = outdir / 'latent.png'

plt.figure(figsize=(11, 4))
plt.plot(hours, glucose, label='Observed CGM')
plt.plot(hours, pred, label='Fitted glucose')
plt.xlabel('Time (hours)')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.tight_layout()
plt.savefig(fig1, dpi=160)
plt.close()

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

print(fig1)
print(fig2)