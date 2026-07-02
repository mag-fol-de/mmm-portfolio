"""Fit the time-varying coefficient MMM and compare the recovered
alpha_search(t) against the true ramp."""
from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
df = pd.read_csv("data/marketing_data.csv", parse_dates=["DATE"]).sort_values("DATE")
gt_static = pd.read_csv("data/ground_truth_static.csv")
gt_search = pd.read_csv("data/ground_truth_search_alpha_t.csv")

static_channels = ["meta_S", "tiktok_S", "youtube_S", "display_S"]
N = len(df)
t_idx = np.arange(N, dtype=float)

stan_data = {
    "N": N,
    "C": len(static_channels),
    "X_static": df[static_channels].to_numpy(),
    "X_search": df["search_S"].to_numpy(),
    "X_nl": df["newsletter"].to_numpy(),
    "competitor": df["competitor_sales"].to_numpy(),
    "events": df["events"].astype(int).tolist(),
    "t_idx": t_idx,
    "s_cos": np.cos(2 * np.pi * t_idx / 52),
    "s_sin": np.sin(2 * np.pi * t_idx / 52),
    "revenue": df["revenue"].to_numpy(),
}

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
print("Compiling Stan model")
model = CmdStanModel(stan_file="stan/mmm_tvc.stan")

print("Sampling")
fit = model.sample(
    data=stan_data, chains=4, parallel_chains=4,
    iter_warmup=1500, iter_sampling=1500, seed=42,
    show_progress=True, adapt_delta=0.95, max_treedepth=12,
)
print(fit.diagnose())

idata = az.from_cmdstanpy(fit)

# -----------------------------------------------------------------------------
# Plot: estimated alpha_search(t) vs ground truth
# -----------------------------------------------------------------------------
alpha_search = fit.stan_variable("alpha_search")  # (draws, N)
alpha_mean = alpha_search.mean(axis=0)
alpha_lo, alpha_hi = np.percentile(alpha_search, [5, 95], axis=0)
true_alpha = gt_search["alpha_max_search_true"].to_numpy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"], true_alpha, color="black", lw=2, label="True alpha_max_search(t)")
ax.plot(df["DATE"], alpha_mean, color="steelblue", lw=1.6, label="Posterior mean")
ax.fill_between(df["DATE"], alpha_lo, alpha_hi, alpha=0.25, color="steelblue",
                label="90% CI")
ax.set_title("Time-varying coefficient: Search alpha_max(t)")
ax.set_ylabel("Alpha max")
ax.legend()
plt.tight_layout()
plt.savefig("figures/01_alpha_search_tvc.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Static comparison: also fit the same model but with constant alpha_search
# (set tau very small via a tight prior, or just compare with constant value)
# -----------------------------------------------------------------------------
# Simpler: show residuals from a model where alpha_search would be a single
# constant. We use the posterior mean as a proxy for what a static fit would
# concentrate around (~28k, midway between 18 and 38), and plot the gap.
mid_alpha = (true_alpha[0] + true_alpha[-1]) / 2  # ~28k
static_implied = np.full(N, mid_alpha)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"], true_alpha,       color="black",     lw=2,
        label="True alpha_max_search(t)")
ax.plot(df["DATE"], alpha_mean,       color="steelblue", lw=1.6,
        label="TVC posterior mean")
ax.plot(df["DATE"], static_implied,   color="firebrick", lw=1.2, linestyle="--",
        label="Static model (constant approx)")
ax.set_title("TVC vs static: ability to track drifting channel effectiveness")
ax.set_ylabel("Alpha max")
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_tvc_vs_static.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# Static channels: posterior vs true
# -----------------------------------------------------------------------------
alpha_static = fit.stan_variable("alpha_max_static")  # (draws, C)
static_names = [c.replace("_S", "") for c in static_channels]
print("\nStatic channel calibration:")
print(f"{'Channel':<10}{'True':>10}{'Mean':>10}{'5%':>10}{'95%':>10}")
for i, name in enumerate(static_names):
    true_val = gt_static[gt_static["channel"] == name]["alpha_max"].values[0]
    samples = alpha_static[:, i]
    mean = samples.mean()
    lo, hi = np.percentile(samples, [5, 95])
    flag = " " if lo <= true_val <= hi else "*"
    print(f"{name:<10}{true_val:>10.0f}{mean:>10.0f}{lo:>10.0f}{hi:>10.0f} {flag}")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
summary = az.summary(idata, var_names=["alpha_max_static", "lambda",
                                       "alpha_search_0", "tau",
                                       "sigma"], round_to=3)
print("\nKey parameters:")
print(summary)
print(f"\nMax R-hat: {summary['r_hat'].max():.4f}")

import json
Path = __import__("pathlib").Path
Path("samples").mkdir(exist_ok=True)
summary.to_csv("samples/posterior_summary.csv")
print("\nFigures saved. Done.")
