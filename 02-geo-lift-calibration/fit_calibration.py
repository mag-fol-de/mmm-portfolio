"""Two-step MMM calibration pipeline.

Step 1: fit a Bayesian geo-lift model to estimate Meta's incremental ROAS.
Step 2: translate that posterior into an informative prior on Meta's alpha_max
        in the MMM, refit with and without the calibration prior, and compare.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CMDSTAN", r"D:\cmdstan\cmdstan-2.39.0")
_TBB = r"D:\cmdstan\cmdstan-2.39.0\stan\lib\stan_math\lib\tbb"
_RT_USR = r"D:\rtools40\usr\bin"
_RT_MGW = r"D:\rtools40\mingw64\bin"
_extra = ";".join([_TBB, _RT_USR, _RT_MGW])
if _extra not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _extra + ";" + os.environ.get("PATH", "")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "samples"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
geo = pd.read_csv(ROOT / "data" / "geo_experiment.csv")
mmm = pd.read_csv(ROOT / "data" / "marketing_data.csv",
                  parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
gt = pd.read_csv(ROOT / "data" / "ground_truth.csv")
exp_truth = pd.read_csv(ROOT / "data" / "experiment_truth.csv").set_index("param")["value"]

true_lift = float(exp_truth["incremental_roas_meta"])
true_meta_alpha = float(exp_truth["true_meta_alpha_max"])
print(f"True incremental ROAS Meta: {true_lift:.3f}")
print(f"True Meta alpha_max: {true_meta_alpha:,.0f}")

# -----------------------------------------------------------------------------
# Step 1: Bayesian lift estimation
# -----------------------------------------------------------------------------
geo["region_idx"] = geo["region"].map({"A": 1, "B": 2}).astype(int)
lift_data = {
    "N": len(geo),
    "R": 2,
    "region": geo["region_idx"].tolist(),
    "baseline_meta": (geo["meta_spend"] - geo["meta_boost"]).to_numpy(),
    "extra_meta": geo["meta_boost"].to_numpy(),
    "revenue": geo["revenue"].to_numpy(),
}

print("\nFitting geo-lift Stan model...")
lift_model = CmdStanModel(stan_file=str(ROOT / "stan" / "lift.stan"))
lift_fit = lift_model.sample(data=lift_data, chains=4, parallel_chains=4,
                              iter_warmup=1000, iter_sampling=1000, seed=11,
                              show_progress=False)
print(lift_fit.diagnose())

lift_samples = lift_fit.stan_variable("lift")
lift_mean = float(lift_samples.mean())
lift_sd = float(lift_samples.std())
lift_lo, lift_hi = np.percentile(lift_samples, [5, 95])
print(f"\nIncremental ROAS Meta posterior:")
print(f"  Mean: {lift_mean:.3f}, 90% CI [{lift_lo:.3f}, {lift_hi:.3f}]")
print(f"  Truth: {true_lift:.3f}")

# Translate lift into informative prior on Meta's alpha_max
total_meta_spend_per_week = float(mmm["meta_S"].mean())
prior_alpha_max_meta_mean = lift_mean * total_meta_spend_per_week * 1.4
prior_alpha_max_meta_sd = lift_sd * total_meta_spend_per_week * 3.0
print(f"\nDerived prior on Meta alpha_max: "
      f"Normal({prior_alpha_max_meta_mean:,.0f}, {prior_alpha_max_meta_sd:,.0f})")

# -----------------------------------------------------------------------------
# Step 2: MMM with calibrated prior + uncalibrated baseline
# -----------------------------------------------------------------------------
paid_channels = ["search_S", "meta_S", "tiktok_S", "youtube_S", "display_S"]
channel_names = [c.replace("_S", "") for c in paid_channels]
META_IDX = channel_names.index("meta") + 1
N = len(mmm)
t_idx = np.arange(N, dtype=float)

mmm_data = {
    "N": N,
    "C": len(paid_channels),
    "X_paid": mmm[paid_channels].to_numpy(),
    "X_nl": mmm["newsletter"].to_numpy(),
    "competitor": mmm["competitor_sales"].to_numpy(),
    "events": mmm["events"].astype(int).tolist(),
    "t_idx": t_idx.tolist(),
    "s_cos": np.cos(2 * np.pi * t_idx / 52).tolist(),
    "s_sin": np.sin(2 * np.pi * t_idx / 52).tolist(),
    "revenue": mmm["revenue"].to_numpy(),
    "meta_idx": META_IDX,
    "prior_alpha_max_meta_mean": float(prior_alpha_max_meta_mean),
    "prior_alpha_max_meta_sd": float(prior_alpha_max_meta_sd),
}

print("\nFitting calibrated MMM...")
mmm_model = CmdStanModel(stan_file=str(ROOT / "stan" / "mmm_calibrated.stan"))
mmm_fit = mmm_model.sample(data=mmm_data, chains=4, parallel_chains=4,
                            iter_warmup=1000, iter_sampling=1000, seed=42,
                            show_progress=False, adapt_delta=0.95,
                            max_treedepth=12)
print(mmm_fit.diagnose())

print("\nFitting uncalibrated MMM (baseline)...")
mmm_data_uncal = {**mmm_data,
                   "prior_alpha_max_meta_mean": 30000.0,
                   "prior_alpha_max_meta_sd": 30000.0}
mmm_fit_uncal = mmm_model.sample(data=mmm_data_uncal, chains=4, parallel_chains=4,
                                  iter_warmup=1000, iter_sampling=1000, seed=42,
                                  show_progress=False, adapt_delta=0.95,
                                  max_treedepth=12)
print(mmm_fit_uncal.diagnose())

# -----------------------------------------------------------------------------
# Compare Meta posterior
# -----------------------------------------------------------------------------
meta_ix = channel_names.index("meta")
meta_cal = mmm_fit.stan_variable("alpha_max")[:, meta_ix]
meta_uncal = mmm_fit_uncal.stan_variable("alpha_max")[:, meta_ix]

def summarize(samples, label):
    mean = float(samples.mean())
    lo, hi = np.percentile(samples, [5, 95])
    width = hi - lo
    inside = lo <= true_meta_alpha <= hi
    return dict(label=label, mean=mean, p05=float(lo), p95=float(hi),
                width=float(width), contains_truth=bool(inside))

results = [
    summarize(meta_uncal, "Uncalibrated"),
    summarize(meta_cal,   "Calibrated"),
]
print("\nMeta alpha_max posterior comparison:")
print(f"  Truth: {true_meta_alpha:,.0f}\n")
print(f"  {'Model':<14} {'Mean':>10} {'5%':>10} {'95%':>10} {'CI width':>12} Contains truth")
print("  " + "-" * 68)
for r in results:
    flag = "yes" if r["contains_truth"] else "NO"
    print(f"  {r['label']:<14} {r['mean']:>10,.0f} {r['p05']:>10,.0f} {r['p95']:>10,.0f}"
          f" {r['width']:>12,.0f}  {flag}")

pd.DataFrame(results + [{
    "label": "Truth", "mean": true_meta_alpha, "p05": None, "p95": None,
    "width": None, "contains_truth": None,
}]).to_csv(OUT / "meta_calibration_comparison.csv", index=False)

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(meta_uncal, bins=40, alpha=0.55, label="Uncalibrated", color="steelblue", density=True)
ax.hist(meta_cal,   bins=40, alpha=0.55, label="Calibrated",   color="darkorange", density=True)
ax.axvline(true_meta_alpha, color="black", linestyle="--", lw=1.5,
           label=f"Truth ({true_meta_alpha:,.0f})")
ax.set_title("Meta alpha_max posterior: calibrated vs uncalibrated")
ax.set_xlabel("alpha_max (SEK)")
ax.set_ylabel("Posterior density")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "01_meta_calibration.png"); plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(lift_samples, bins=40, color="seagreen", alpha=0.8, density=True)
ax.axvline(true_lift, color="black", linestyle="--", lw=1.5,
           label=f"Truth ({true_lift:.3f})")
ax.set_title("Posterior of incremental Meta ROAS from geo-lift")
ax.set_xlabel("Incremental ROAS")
ax.set_ylabel("Posterior density")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "02_lift_posterior.png"); plt.close()

print(f"\nFigures written to {FIG}")
print(f"Tables written to {OUT}")
print("Done.")
