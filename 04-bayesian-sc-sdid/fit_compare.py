"""Fit Bayesian SC and Bayesian SDID with Stan (HMC-NUTS), compare ATT
posteriors against the known ground truth, and produce diagnostic plots.

Sampler: Stan's HMC-NUTS via cmdstanpy. The Stan files in stan/ define
the canonical model specification.
"""
from __future__ import annotations

import os
from pathlib import Path

# Point cmdstanpy at the D-drive cmdstan build and the RTools40 toolchain
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
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "samples"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SEED = 42

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
Y_wide = pd.read_csv(ROOT / "data" / "Y_wide.csv").drop(columns=["week"])
gt = pd.read_csv(ROOT / "data" / "ground_truth.csv").set_index("param")
TRUE_TAU = float(gt.loc["true_tau", "value"])
T0 = int(gt.loc["T0", "value"])
T = int(gt.loc["T", "value"])
TREATED = int(gt.loc["treated_unit", "value"])

Y_treated = Y_wide[f"unit_{TREATED}"].to_numpy()
control_cols = [c for c in Y_wide.columns if c != f"unit_{TREATED}"]
Y_control = Y_wide[control_cols].to_numpy()                       # (T, N)
N = Y_control.shape[1]

stan_data = {
    "T": T,
    "T0": T0,
    "N": N,
    "Y_treated": Y_treated.tolist(),
    "Y_control": Y_control.tolist(),
}

print(f"True tau: {TRUE_TAU:,.0f}")
print(f"T={T}, T0={T0}, N controls={N}")

# -----------------------------------------------------------------------------
# Fit Bayesian SC
# -----------------------------------------------------------------------------
print("\nCompiling and fitting Bayesian SC (Stan HMC-NUTS)...")
sc_model = CmdStanModel(stan_file=str(ROOT / "stan" / "bayesian_sc.stan"))
sc_fit = sc_model.sample(
    data=stan_data, chains=4, parallel_chains=4,
    iter_warmup=1500, iter_sampling=1500, seed=SEED,
    show_progress=False, adapt_delta=0.95,
)
print(sc_fit.diagnose())

sc_att = sc_fit.stan_variable("att_mean")
sc_att_t = sc_fit.stan_variable("att_t")          # (draws, T - T0)
sc_omega = sc_fit.stan_variable("omega")          # (draws, N)
sc_cf = sc_fit.stan_variable("counterfactual")    # (draws, T)

# -----------------------------------------------------------------------------
# Fit Bayesian SDID
# -----------------------------------------------------------------------------
print("\nCompiling and fitting Bayesian SDID (Stan HMC-NUTS)...")
sdid_model = CmdStanModel(stan_file=str(ROOT / "stan" / "bayesian_sdid.stan"))
sdid_fit = sdid_model.sample(
    data=stan_data, chains=4, parallel_chains=4,
    iter_warmup=1500, iter_sampling=1500, seed=SEED,
    show_progress=False, adapt_delta=0.95,
)
print(sdid_fit.diagnose())

sdid_att = sdid_fit.stan_variable("att_mean")
sdid_att_t = sdid_fit.stan_variable("att_t")
sdid_omega = sdid_fit.stan_variable("omega")
sdid_lambda = sdid_fit.stan_variable("lambda")
sdid_cf = sdid_fit.stan_variable("counterfactual")

# -----------------------------------------------------------------------------
# Compare ATT posteriors
# -----------------------------------------------------------------------------
def summarize(samples, label):
    mean = samples.mean()
    lo, hi = np.percentile(samples, [5, 95])
    inside = lo <= TRUE_TAU <= hi
    width = hi - lo
    bias = mean - TRUE_TAU
    print(f"  {label:<6} mean={mean:>10,.0f}  90% CI [{lo:>10,.0f}, {hi:>10,.0f}]"
          f"  width={width:>10,.0f}  bias={bias:>+10,.0f}  contains truth: {'yes' if inside else 'NO'}")
    return dict(mean=float(mean), p05=float(lo), p95=float(hi),
                width=float(width), bias=float(bias), contains_truth=bool(inside))


print("\nATT posterior summary:")
print(f"  Truth: {TRUE_TAU:,.0f}")
sc_stats = summarize(sc_att, "SC")
sdid_stats = summarize(sdid_att, "SDID")

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

# 0. Raw panel: treated vs controls over time (so the reader sees the data)
fig, ax = plt.subplots(figsize=(11, 5))
weeks = np.arange(T)
for j in range(N):
    ax.plot(weeks, Y_control[:, j], color="lightgrey", lw=0.7, alpha=0.7)
ax.plot([], [], color="lightgrey", lw=0.7, label=f"{N} control markets")
ax.plot(weeks, Y_treated, color="darkorange", lw=2.0, label="Treated market")
ax.axvline(T0 - 0.5, color="red", linestyle="--", lw=1.2, label="Treatment start")
ax.set_xlabel("Week")
ax.set_ylabel("Weekly outcome (SEK)")
ax.set_title("Panel data: treated market vs controls over time")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "00_panel_overview.png")
plt.close()

# 1. ATT posterior plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(sc_att,   bins=40, alpha=0.55, color="steelblue",  label="Bayesian SC", density=True)
ax.hist(sdid_att, bins=40, alpha=0.55, color="darkorange", label="Bayesian SDID", density=True)
ax.axvline(TRUE_TAU, color="black", linestyle="--", lw=1.5,
           label=f"Truth ({TRUE_TAU:,.0f})")
ax.set_title("ATT posterior: Bayesian SC vs Bayesian SDID")
ax.set_xlabel("ATT")
ax.set_ylabel("Posterior density")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "01_att_posteriors.png")
plt.close()


def plot_counterfactual(cf_draws, label, fname, color):
    cf_mean = cf_draws.mean(axis=0)
    cf_lo, cf_hi = np.percentile(cf_draws, [5, 95], axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    weeks = np.arange(T)
    ax.plot(weeks, Y_treated, color="black", lw=2, label="Treated unit (observed)")
    ax.plot(weeks, cf_mean, color=color, lw=1.5, label=f"{label} counterfactual (mean)")
    ax.fill_between(weeks, cf_lo, cf_hi, alpha=0.25, color=color, label="90% CI")
    ax.axvline(T0 - 0.5, color="grey", linestyle="--", lw=1, label="Treatment start")
    ax.set_title(f"{label}: treated vs synthetic counterfactual")
    ax.set_xlabel("Week")
    ax.set_ylabel("Outcome")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


plot_counterfactual(sc_cf, "Bayesian SC",
                    FIG_DIR / "02_counterfactual_sc.png", "steelblue")
plot_counterfactual(sdid_cf, "Bayesian SDID",
                    FIG_DIR / "03_counterfactual_sdid.png", "darkorange")

# 4. Unit weights
sc_w = sc_omega.mean(axis=0)
sdid_w = sdid_omega.mean(axis=0)
unit_labels = list(control_cols)
fig, ax = plt.subplots(figsize=(11, 5))
idx = np.arange(N)
width = 0.4
ax.bar(idx - width / 2, sc_w,   width=width, color="steelblue",  label="SC weights")
ax.bar(idx + width / 2, sdid_w, width=width, color="darkorange", label="SDID weights")
ax.set_xticks(idx)
ax.set_xticklabels(unit_labels, rotation=45, ha="right", fontsize=8)
ax.set_title("Unit weights: SC vs SDID (posterior mean)")
ax.set_ylabel("omega")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "04_unit_weights.png")
plt.close()

# 5. SDID time weights
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(np.arange(T0), sdid_lambda.mean(axis=0), color="darkorange")
ax.set_title("SDID time weights over the pre-period (posterior mean lambda)")
ax.set_xlabel("Pre-period week")
ax.set_ylabel("lambda")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_time_weights.png")
plt.close()

# 6. ATT path per post-period week
fig, ax = plt.subplots(figsize=(11, 5))
weeks_post = np.arange(T0, T)
sc_att_mean = sc_att_t.mean(axis=0)
sc_att_lo, sc_att_hi = np.percentile(sc_att_t, [5, 95], axis=0)
sdid_att_mean = sdid_att_t.mean(axis=0)
sdid_att_lo, sdid_att_hi = np.percentile(sdid_att_t, [5, 95], axis=0)
ax.plot(weeks_post, sc_att_mean, color="steelblue", lw=2, label="BSC ATT (mean)")
ax.fill_between(weeks_post, sc_att_lo, sc_att_hi, color="steelblue", alpha=0.2)
ax.plot(weeks_post, sdid_att_mean, color="darkorange", lw=2, label="BSDID ATT (mean)")
ax.fill_between(weeks_post, sdid_att_lo, sdid_att_hi, color="darkorange", alpha=0.2)
ax.axhline(TRUE_TAU, color="black", linestyle="--", lw=1, label=f"Truth ({TRUE_TAU:,.0f})")
ax.set_xlabel("Week")
ax.set_ylabel("ATT")
ax.set_title("Per-week ATT posterior with 90% CI, post-period")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "06_att_path.png")
plt.close()

# -----------------------------------------------------------------------------
# Save summaries
# -----------------------------------------------------------------------------
pd.DataFrame({
    "method": ["SC", "SDID"],
    "true_tau": [TRUE_TAU, TRUE_TAU],
    "att_mean":  [sc_stats["mean"], sdid_stats["mean"]],
    "att_p05":   [sc_stats["p05"],  sdid_stats["p05"]],
    "att_p95":   [sc_stats["p95"],  sdid_stats["p95"]],
    "ci_width":  [sc_stats["width"], sdid_stats["width"]],
    "bias":      [sc_stats["bias"],  sdid_stats["bias"]],
    "contains_truth": [sc_stats["contains_truth"], sdid_stats["contains_truth"]],
}).to_csv(OUT_DIR / "att_comparison.csv", index=False)

print(f"\nWrote figures to {FIG_DIR}")
print(f"Wrote summary to {OUT_DIR / 'att_comparison.csv'}")
print("Done.")
