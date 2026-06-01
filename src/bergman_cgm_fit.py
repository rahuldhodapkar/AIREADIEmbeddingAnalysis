#!/usr/bin/env python
"""Semi-mechanistic CGM fitting with a Bergman-style ODE.

This script is designed for CGM-only traces. A strict Bergman model often fits
poorly without insulin measurements, so this version adds two pragmatic pieces:

1) a CGM sensor lag state, and
2) a smooth latent disturbance term that can absorb unobserved inputs.

It still keeps a mechanistic core, but the model is deliberately flexible enough
that the fitted curve should usually track CGM substantially better.

Model
-----

Blood glucose dynamics:

    dG/dt = -(p1 + X) * (G - Gb) + sg * M + D(t)

Insulin-action proxy:

    dX/dt = -p2 * X + p3 * max(0, M)

Meal absorption chain:

    dQ/dt = -kq * Q + u_meal(t)
    dM/dt = -km * M + kq * Q

CGM sensor lag:

    dY/dt = (G - Y) / tau_cgm

Observed data are compared against Y(t), not G(t).

Here:
- G is blood glucose
- Y is CGM sensor glucose
- X is a latent insulin-action proxy
- Q and M form a two-stage meal absorption chain
- u_meal(t) is a nonnegative meal drive represented by Gaussian basis functions
- D(t) is a signed smooth disturbance term represented by Gaussian basis functions

Usage
-----
Fit real CGM:
    python bergman_cgm_fit.py --csv cgm.csv

Simulate synthetic data and fit it:
    python bergman_cgm_fit.py --simulate --save-plot synthetic_fit.png

Optional:
    --noise-sd 8.0
    --n-minutes 360
    --dt 5
    --restarts 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


# -----------------------------
# Utilities
# -----------------------------

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-8)
    return np.log(np.expm1(y)) if y < 50 else y


def gaussian_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    t = np.asarray(t)[:, None]
    c = np.asarray(centers)[None, :]
    return np.exp(-0.5 * ((t - c) / width) ** 2)


def smooth_glucose_for_seed(g: np.ndarray) -> np.ndarray:
    n = len(g)
    if n < 7:
        return g
    win = min(n if n % 2 == 1 else n - 1, 11)
    if win < 5:
        return g
    return savgol_filter(g, window_length=win, polyorder=2, mode="interp")


def finite_diff_penalty(vec: np.ndarray) -> np.ndarray:
    if len(vec) < 3:
        return np.zeros(0, dtype=float)
    return np.diff(vec, n=2)


@dataclass
class FitConfig:
    meal_centers: np.ndarray
    dist_centers: np.ndarray
    basis_width_min: float
    x0_glucose: float
    x0_x: float = 0.0
    x0_q: float = 0.0
    x0_m: float = 0.0
    x0_y: float = 0.0
    lambda_meal: float = 0.08
    lambda_dist: float = 0.04
    lambda_smooth: float = 0.35
    lambda_tau: float = 0.02


# -----------------------------
# Model definition
# -----------------------------

def unpack_positive(theta_raw: np.ndarray, n: int) -> np.ndarray:
    return softplus(theta_raw[:n]) + 1e-8


def build_latent_drives(
    t_eval: np.ndarray,
    theta: np.ndarray,
    config: FitConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build meal drive u_meal(t) and signed disturbance D(t)."""
    n_meal = len(config.meal_centers)
    n_dist = len(config.dist_centers)

    # Positive parameters first, including tau_cgm.
    p1, p2, p3, kq, km, sg, Gb, tau_cgm = unpack_positive(theta, 8)

    idx = 8
    meal_raw = theta[idx : idx + n_meal]
    idx += n_meal
    dist_raw = theta[idx : idx + n_dist]

    meal_coef = softplus(meal_raw)
    dist_coef = dist_raw.copy()  # signed coefficients allowed

    B_meal = gaussian_basis(t_eval, config.meal_centers, config.basis_width_min)
    B_dist = gaussian_basis(t_eval, config.dist_centers, config.basis_width_min)

    u_meal = B_meal @ meal_coef
    disturbance = B_dist @ dist_coef

    params = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "kq": kq,
        "km": km,
        "sg": sg,
        "Gb": Gb,
        "tau_cgm": tau_cgm,
        "meal_coef": meal_coef,
        "dist_coef": dist_coef,
    }
    return u_meal, disturbance, params


def simulate_model(
    t_eval: np.ndarray,
    theta: np.ndarray,
    config: FitConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Simulate glucose, sensor glucose, and latent states."""
    u_meal, disturbance, params = build_latent_drives(t_eval, theta, config)

    p1 = params["p1"]
    p2 = params["p2"]
    p3 = params["p3"]
    kq = params["kq"]
    km = params["km"]
    sg = params["sg"]
    Gb = params["Gb"]
    tau_cgm = params["tau_cgm"]

    def u_meal_t(t: float) -> float:
        return float(np.interp(t, t_eval, u_meal))

    def d_t(t: float) -> float:
        return float(np.interp(t, t_eval, disturbance))

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        G, X, Q, M, Y = y
        meal_drive = u_meal_t(t)
        D = d_t(t)
        dG = -(p1 + X) * (G - Gb) + sg * M + D
        dX = -p2 * X + p3 * max(0.0, M)
        dQ = -kq * Q + meal_drive
        dM = -km * M + kq * Q
        dY = (G - Y) / tau_cgm
        return np.array([dG, dX, dQ, dM, dY])

    y0 = np.array(
        [config.x0_glucose, config.x0_x, config.x0_q, config.x0_m, config.x0_y],
        dtype=float,
    )

    sol = solve_ivp(
        ode,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    G, X, Q, M, Y = sol.y
    return G, X, Q, M, Y, params


# -----------------------------
# Synthetic-data generation
# -----------------------------

def make_synthetic_theta(meal_centers: np.ndarray, dist_centers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a plausible parameter vector for simulation.

    The synthetic day is built to look like a typical CGM trace with distinct
    breakfast, lunch, and dinner excursions rather than a nearly flat curve.
    """
    p1 = inv_softplus(0.020)
    p2 = inv_softplus(0.050)
    p3 = inv_softplus(0.018)
    kq = inv_softplus(0.060)
    km = inv_softplus(0.035)
    sg = inv_softplus(1.100)
    Gb = inv_softplus(95.0)
    tau_cgm = inv_softplus(10.0)

    meal_coef_true = np.zeros(len(meal_centers), dtype=float)
    # Use a small number of explicit meal pulses at realistic times.
    # These are translated into the latent meal drive through the Gaussian basis.
    meal_hours = np.array([8.0, 13.0, 19.0])
    meal_times = meal_hours * 60.0
    meal_amps = np.array([5.0, 6.0, 5.5])
    for tm, amp in zip(meal_times, meal_amps):
        j = int(np.argmin(np.abs(meal_centers - tm)))
        meal_coef_true[j] = amp

    # Small background meal activity so the fit does not become unrealistically sparse.
    meal_coef_true += 0.05 * rng.random(len(meal_centers))

    # Disturbance is kept modest so the synthetic curve remains physiologically plausible.
    dist_coef_true = np.zeros(len(dist_centers), dtype=float)
    if len(dist_centers) > 0:
        morning = int(np.argmin(np.abs(dist_centers - 150.0)))
        afternoon = int(np.argmin(np.abs(dist_centers - 420.0)))
        dist_coef_true[morning] = 0.4
        dist_coef_true[afternoon] = -0.3

    theta = np.concatenate(
        [
            np.array([p1, p2, p3, kq, km, sg, Gb, tau_cgm]),
            np.log(np.expm1(np.maximum(meal_coef_true, 1e-6))),
            dist_coef_true,
        ]
    )
    return theta


def simulate_synthetic_cgm(
    n_minutes: int = 360,
    dt: float = 5.0,
    noise_sd: float = 8.0,
    seed: int = 0,
) -> dict:
    """Generate synthetic CGM data using the same ODE structure as the fit.

    The synthetic trace is designed to look more like a typical day:
    - distinct meal excursions,
    - moderate CGM lag,
    - mild unmeasured disturbance,
    - non-flat glucose dynamics.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, float(n_minutes) + 1e-9, float(dt))

    spacing = 15.0
    n_basis = max(5, int(np.ceil(n_minutes / spacing)) + 1)
    meal_centers = np.linspace(t[0], t[-1], n_basis)
    dist_centers = np.linspace(t[0], t[-1], n_basis)

    config = FitConfig(
        meal_centers=meal_centers,
        dist_centers=dist_centers,
        basis_width_min=max(8.0, spacing),
        x0_glucose=95.0,
        x0_x=0.0,
        x0_q=0.0,
        x0_m=0.0,
        x0_y=95.0,
    )

    theta_true = make_synthetic_theta(meal_centers, dist_centers, rng)
    G, X, Q, M, Y, params_true = simulate_model(t, theta_true, config)
    glucose_obs = Y + rng.normal(0.0, noise_sd, size=Y.shape)

    u_meal_true, disturbance_true, _ = build_latent_drives(t, theta_true, config)

    return {
        "t": t,
        "glucose_clean": Y,
        "glucose_obs": glucose_obs,
        "G_clean": G,
        "X_clean": X,
        "Q_clean": Q,
        "M_clean": M,
        "u_meal_true": u_meal_true,
        "disturbance_true": disturbance_true,
        "theta_true": theta_true,
        "params_true": params_true,
        "config": config,
        "noise_sd": noise_sd,
    }


# -----------------------------
# Initialization helpers
# -----------------------------

def estimate_initial_meal_coeffs(t_eval: np.ndarray, glucose_obs: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    g_smooth = smooth_glucose_for_seed(glucose_obs)
    if len(t_eval) < 3:
        return np.full(len(centers), inv_softplus(0.1))
    dgdt = np.gradient(g_smooth, t_eval)
    target = np.maximum(dgdt, 0.0)
    B = gaussian_basis(t_eval, centers, width)
    coef, *_ = np.linalg.lstsq(B, target, rcond=None)
    coef = np.clip(coef, 0.0, None)
    coef = 0.1 + coef
    return np.array([inv_softplus(c) for c in coef])


def make_initial_theta(t_eval: np.ndarray, glucose_obs: np.ndarray, config: FitConfig, rng: np.random.Generator) -> np.ndarray:
    p1_0 = inv_softplus(0.02)
    p2_0 = inv_softplus(0.05)
    p3_0 = inv_softplus(0.02)
    kq_0 = inv_softplus(0.06)
    km_0 = inv_softplus(0.03)
    sg_0 = inv_softplus(1.0)
    Gb_0 = inv_softplus(max(60.0, np.percentile(glucose_obs, 20)))
    tau_0 = inv_softplus(12.0)

    meal_seed = estimate_initial_meal_coeffs(t_eval, glucose_obs, config.meal_centers, config.basis_width_min)
    meal_seed = meal_seed + 0.15 * rng.standard_normal(len(config.meal_centers))
    dist_seed = 0.02 * rng.standard_normal(len(config.dist_centers))

    return np.concatenate(
        [
            np.array([p1_0, p2_0, p3_0, kq_0, km_0, sg_0, Gb_0, tau_0]),
            meal_seed,
            dist_seed,
        ]
    )


# -----------------------------
# Objective and fitting
# -----------------------------

def residuals(theta: np.ndarray, t_eval: np.ndarray, glucose_obs: np.ndarray, config: FitConfig, glucose_scale: float) -> np.ndarray:
    _, _, _, _, Y_hat, params = simulate_model(t_eval, theta, config)
    resid = (Y_hat - glucose_obs) / glucose_scale

    u_meal, disturbance, _ = build_latent_drives(t_eval, theta, config)
    meal_coef = params["meal_coef"]
    dist_coef = params["dist_coef"]

    reg = []
    reg.append(np.sqrt(config.lambda_meal) * meal_coef)
    reg.append(np.sqrt(config.lambda_dist) * dist_coef)
    reg.append(np.sqrt(config.lambda_smooth) * finite_diff_penalty(meal_coef))
    reg.append(np.sqrt(config.lambda_smooth) * finite_diff_penalty(dist_coef))
    reg.append(np.array([np.sqrt(config.lambda_tau) * np.log(params["tau_cgm"]) ]))

    return np.concatenate([resid] + [r.ravel() for r in reg if len(r) > 0])


def fit_cgm_trace(t_eval: np.ndarray, glucose_obs: np.ndarray, config: FitConfig, random_seed: int = 0, restarts: int = 5) -> dict:
    rng = np.random.default_rng(random_seed)
    glucose_scale = max(10.0, float(np.std(glucose_obs)))

    theta_base = make_initial_theta(t_eval, glucose_obs, config, rng)
    lb = np.full(theta_base.shape, -1e6)
    ub = np.full(theta_base.shape, 1e6)

    best_result = None
    best_cost = np.inf

    for k in range(max(1, int(restarts))):
        theta0 = theta_base.copy()
        if k > 0:
            theta0[:8] += rng.normal(0.0, 0.2, size=8)
            theta0[8:] += rng.normal(0.0, 0.5, size=len(theta0) - 8)

        result = least_squares(
            residuals,
            theta0,
            bounds=(lb, ub),
            args=(t_eval, glucose_obs, config, glucose_scale),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=1500,
            verbose=0,
        )

        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    assert best_result is not None
    G_hat, X_hat, Q_hat, M_hat, Y_hat, params = simulate_model(t_eval, best_result.x, config)
    u_meal, disturbance, _ = build_latent_drives(t_eval, best_result.x, config)

    return {
        "result": best_result,
        "params": params,
        "t": t_eval,
        "glucose_obs": glucose_obs,
        "glucose_fit": Y_hat,
        "G_fit": G_hat,
        "X_fit": X_hat,
        "Q_fit": Q_hat,
        "M_fit": M_hat,
        "u_meal_fit": u_meal,
        "disturbance_fit": disturbance,
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_fit(fit: dict, out_path: Path | None = None, truth: dict | None = None) -> None:
    t = fit["t"]
    glucose_obs = fit["glucose_obs"]
    glucose_fit = fit["glucose_fit"]
    u_meal = fit["u_meal_fit"]
    disturbance = fit["disturbance_fit"]
    X_fit = fit["X_fit"]
    Q_fit = fit["Q_fit"]
    M_fit = fit["M_fit"]
    params = fit["params"]

    nrows = 4 if truth is not None else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4.0 * nrows), sharex=True)
    if nrows == 3:
        axes = np.asarray(axes)

    axes[0].plot(t, glucose_obs, "o", ms=3, alpha=0.7, label="Observed CGM")
    axes[0].plot(t, glucose_fit, lw=2, label="Fitted CGM")
    if truth is not None:
        axes[0].plot(t, truth["glucose_clean"], lw=2, linestyle="--", label="True clean CGM")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].legend()
    axes[0].set_title("Semi-mechanistic CGM fit")

    axes[1].plot(t, u_meal, lw=2, label="Latent meal drive u_meal(t)")
    axes[1].plot(t, Q_fit, lw=2, label="Gut state Q(t)")
    axes[1].plot(t, M_fit, lw=2, label="Appearance state M(t)")
    if truth is not None:
        axes[1].plot(t, truth["u_meal_true"], lw=2, linestyle="--", label="True meal drive")
        axes[1].plot(t, truth["Q_clean"], lw=2, linestyle="--", label="True Q(t)")
        axes[1].plot(t, truth["M_clean"], lw=2, linestyle="--", label="True M(t)")
    axes[1].set_ylabel("Meal signal")
    axes[1].legend(ncol=2)

    axes[2].plot(t, disturbance, lw=2, label="Latent disturbance D(t)")
    axes[2].plot(t, X_fit, lw=2, label="Latent insulin-action proxy X(t)")
    if truth is not None:
        axes[2].plot(t, truth["X_clean"], lw=2, linestyle="--", label="True X(t)")
        axes[2].plot(t, truth["disturbance_true"], lw=2, linestyle="--", label="True disturbance")
    axes[2].axhline(0.0, linestyle="--", linewidth=1)
    axes[2].set_ylabel("Latent states")
    axes[2].legend(ncol=2)

    if nrows == 4:
        axes[3].plot(t, glucose_obs - glucose_fit, lw=1.5, label="Residual")
        axes[3].axhline(0.0, linestyle="--", linewidth=1)
        axes[3].set_ylabel("Obs - fit")
        axes[3].set_xlabel("Time (min)")
        axes[3].legend()
    else:
        axes[2].set_xlabel("Time (min)")

    txt = (
        f"p1={params['p1']:.4g}, p2={params['p2']:.4g}, p3={params['p3']:.4g}, "
        f"kq={params['kq']:.4g}, km={params['km']:.4g}, sg={params['sg']:.4g}, "
        f"Gb={params['Gb']:.2f}, tau={params['tau_cgm']:.2f}"
    )
    fig.text(0.5, 0.01, txt, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


# -----------------------------
# Data loading
# -----------------------------

def load_cgm_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    time_col = None
    glucose_col = None
    for candidate in ["time_minutes", "time", "t", "minutes"]:
        if candidate in cols:
            time_col = cols[candidate]
            break
    for candidate in ["glucose_mgdl", "glucose", "g", "cgm"]:
        if candidate in cols:
            glucose_col = cols[candidate]
            break
    if time_col is None or glucose_col is None:
        raise ValueError("CSV must have time and glucose columns. Expected something like time_minutes and glucose_mgdl.")
    t = df[time_col].to_numpy(dtype=float)
    g = df[glucose_col].to_numpy(dtype=float)
    order = np.argsort(t)
    return t[order], g[order]


def make_default_config(t_eval: np.ndarray, glucose_obs: np.ndarray) -> FitConfig:
    duration = float(t_eval[-1] - t_eval[0])
    spacing = 15.0
    n_basis = max(5, int(np.ceil(duration / spacing)) + 1)
    meal_centers = np.linspace(t_eval[0], t_eval[-1], n_basis)
    dist_centers = np.linspace(t_eval[0], t_eval[-1], n_basis)
    width = max(8.0, spacing)
    g0 = float(glucose_obs[0])
    return FitConfig(
        meal_centers=meal_centers,
        dist_centers=dist_centers,
        basis_width_min=width,
        x0_glucose=g0,
        x0_y=g0,
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CGM CSV")
    parser.add_argument("--simulate", action="store_true", help="Generate synthetic CGM data and fit it")
    parser.add_argument("--n-minutes", type=int, default=1440, help="Synthetic trace length in minutes")
    parser.add_argument("--dt", type=float, default=5.0, help="Synthetic sample spacing in minutes")
    parser.add_argument("--noise-sd", type=float, default=8.0, help="Synthetic CGM noise standard deviation")
    parser.add_argument("--restarts", type=int, default=6, help="Number of optimizer restarts")
    parser.add_argument("--save-plot", type=str, default=None, help="Optional output path for the fit figure")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization and simulation")
    args = parser.parse_args()

    if args.simulate:
        synth = simulate_synthetic_cgm(
            n_minutes=args.n_minutes,
            dt=args.dt,
            noise_sd=args.noise_sd,
            seed=args.seed,
        )
        t = synth["t"]
        g = synth["glucose_obs"]
        config = synth["config"]
        fit = fit_cgm_trace(t, g, config=config, random_seed=args.seed, restarts=args.restarts)
        print("Optimization success:", fit["result"].success)
        print("Cost:", fit["result"].cost)
        print("Message:", fit["result"].message)
        print("Estimated mechanistic parameters:")
        for k, v in fit["params"].items():
            if isinstance(v, np.ndarray):
                continue
            print(f"  {k}: {v:.6g}")
        if args.save_plot is not None:
            plot_fit(fit, Path(args.save_plot), truth=synth)
        else:
            plot_fit(fit, truth=synth)
        return

    if args.csv is None:
        raise SystemExit("Provide --csv path or use --simulate.")

    csv_path = Path(args.csv)
    t, g = load_cgm_csv(csv_path)
    config = make_default_config(t, g)

    fit = fit_cgm_trace(t, g, config=config, random_seed=args.seed, restarts=args.restarts)
    print("Optimization success:", fit["result"].success)
    print("Cost:", fit["result"].cost)
    print("Message:", fit["result"].message)
    print("Estimated mechanistic parameters:")
    for k, v in fit["params"].items():
        if isinstance(v, np.ndarray):
            continue
        print(f"  {k}: {v:.6g}")

    if args.save_plot is not None:
        plot_fit(fit, Path(args.save_plot))
    else:
        plot_fit(fit)


if __name__ == "__main__":
    main()


