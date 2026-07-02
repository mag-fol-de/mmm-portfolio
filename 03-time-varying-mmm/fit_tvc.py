"""Fit the time-varying coefficient MMM and compare the recovered
alpha_search(t) against the true ramp."""
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
df = pd.read_csv(ROOT / "data" / "marketing_data.csv",
                 parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
gt_static = pd.read_csv(ROOT / "data" / "ground_truth_static.csv")
gt_search = pd.read_csv(ROOT / "data" / "ground_truth_search_alpha_t.csv")

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
    "t_idx": t_idx.tolist(),
    "s_cos": np.cos(2 * np.pi * t_idx / 52).tolist(),
    "s_sin": np.sin(2 * np.pi * t_idx / 52).tolist(),
    "revenue": df["revenue"].to_numpy(),
}

# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------
print("Compiling Stan model...")
model = CmdStanModel(stan_file=str(ROOT / "stan" / "mmm_tvc.stan"))

print("Sampling...")
fit = model.sample(
    data=stan_data, chains=4, parallel_chains=4,
    iter_warmup=1500, iter_sampling=1500, seed=42,
    show_progress=False, adapt_delta=0.95, max_treedepth=12,
)
print(fit.diagnose())

# -----------------------------------------------------------------------------
# Alpha_search(t) recovery
# -----------------------------------------------------------------------------
alpha_search = fit.stan_variable("alpha_search")     # (draws, N)
alpha_mean = alpha_search.mean(axis=0)
alpha_lo, alpha_hi = np.percentile(alpha_search, [5, 95], axis=0)
true_alpha = gt_search["alpha_max_search_true"].to_numpy()

# Coverage across time
covers_t = (alpha_lo <= true_alpha) & (true_alpha <= alpha_hi)
coverage_rate = float(covers_t.mean())
mae_alpha = float(np.mean(np.abs(alpha_mean - true_alpha)))
print(f"\nAlpha_search(t) recovery:")
print(f"  Coverage: {coverage_rate:.1%} of weeks contain truth in 90% CI")
print(f"  MAE: {mae_alpha:,.0f}")

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"], true_alpha, color="black", lw=2, label="True alpha_max_search(t)")
ax.plot(df["DATE"], alpha_mean, color="steelblue", lw=1.6, label="Posterior mean")
ax.fill_between(df["DATE"], alpha_lo, alpha_hi, alpha=0.25, color="steelblue",
                label="90% CI")
ax.set_title("Time-varying coefficient: Search alpha_max(t)")
ax.set_ylabel("Alpha max"); ax.set_xlabel("Date")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "01_alpha_search_tvc.png"); plt.close()

# TVC vs static baseline
mid_alpha = float((true_alpha[0] + true_alpha[-1]) / 2)
static_implied = np.full(N, mid_alpha)
static_error = np.abs(static_implied - true_alpha)
tvc_error = np.abs(alpha_mean - true_alpha)
print(f"\nStatic-model implied constant: {mid_alpha:,.0f}")
print(f"  Static MAE (vs true drift): {float(static_error.mean()):,.0f}")
print(f"  TVC MAE:                    {mae_alpha:,.0f}")
print(f"  Ratio: TVC {mae_alpha / float(static_error.mean()):.2%} of static error")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"], true_alpha,     color="black",     lw=2,
        label="True alpha_max_search(t)")
ax.plot(df["DATE"], alpha_mean,     color="steelblue", lw=1.6,
        label="TVC posterior mean")
ax.plot(df["DATE"], static_implied, color="firebrick", lw=1.2, linestyle="--",
        label=f"Static approximation ({mid_alpha:,.0f})")
ax.set_title("TVC vs static: tracking a drifting channel coefficient")
ax.set_ylabel("Alpha max"); ax.set_xlabel("Date")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "02_tvc_vs_static.png"); plt.close()

# -----------------------------------------------------------------------------
# Static channels: parameter recovery
# -----------------------------------------------------------------------------
alpha_static = fit.stan_variable("alpha_max_static")   # (draws, C)
static_names = [c.replace("_S", "") for c in static_channels]

print("\nStatic-channel parameter recovery:")
print(f"{'Channel':<10}{'True':>10}{'Mean':>10}{'5%':>10}{'95%':>10}  Covers")
print("-" * 60)
static_rows = []
for i, name in enumerate(static_names):
    true_val = float(gt_static[gt_static["channel"] == name]["alpha_max"].values[0])
    samples = alpha_static[:, i]
    mean = float(samples.mean())
    lo, hi = np.percentile(samples, [5, 95])
    covers = lo <= true_val <= hi
    flag = "yes" if covers else "NO"
    print(f"{name:<10}{true_val:>10,.0f}{mean:>10,.0f}{lo:>10,.0f}{hi:>10,.0f}  {flag}")
    static_rows.append(dict(channel=name, true=true_val, mean=mean,
                            p05=float(lo), p95=float(hi), covers=covers))

pd.DataFrame(static_rows).to_csv(OUT / "static_recovery.csv", index=False)

pd.Series({
    "alpha_search_coverage_rate": coverage_rate,
    "alpha_search_mae": mae_alpha,
    "static_approx_mae": float(static_error.mean()),
    "tvc_mae_over_static_mae": mae_alpha / float(static_error.mean()),
    "n_weeks": N,
}).to_csv(OUT / "metrics.csv")

print(f"\nFigures written to {FIG}")
print(f"Tables written to {OUT}")
print("Done.")
