"""Two-step calibration pipeline.

Step 1: fit a Bayesian model on the geo-lift experiment to get a posterior on
the incremental ROAS of Meta.

Step 2: convert that posterior into an informative prior on Meta's alpha_max,
fit the MMM with that prior, and compare to an uncalibrated baseline fit.
"""
from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# -----------------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------------
geo = pd.read_csv("data/geo_experiment.csv")
mmm = pd.read_csv("data/marketing_data.csv", parse_dates=["DATE"]).sort_values("DATE")
gt = pd.read_csv("data/ground_truth.csv")
exp_truth = pd.read_csv("data/experiment_truth.csv")

print("True incremental ROAS Meta:",
      exp_truth.set_index("param").loc["incremental_roas_meta", "value"])
print("True Meta alpha_max:",
      exp_truth.set_index("param").loc["true_meta_alpha_max", "value"])

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

print("\nFitting geo-lift Stan model")
lift_model = CmdStanModel(stan_file="stan/lift.stan")
lift_fit = lift_model.sample(data=lift_data, chains=4, parallel_chains=4,
                              iter_warmup=1000, iter_sampling=1000, seed=11,
                              show_progress=True)
print(lift_fit.diagnose())
lift_idata = az.from_cmdstanpy(lift_fit)

lift_summary = az.summary(lift_idata, var_names=["lift", "slope_baseline",
                                                  "mu_region", "sigma"],
                          round_to=3)
print("\nLift posterior summary:")
print(lift_summary)

lift_samples = lift_fit.stan_variable("lift")
lift_mean = lift_samples.mean()
lift_sd = lift_samples.std()
lift_lo, lift_hi = np.percentile(lift_samples, [5, 95])
print(f"\nIncremental ROAS Meta posterior: {lift_mean:.3f} "
      f"(90% CI {lift_lo:.3f} to {lift_hi:.3f})")

# Translate the lift into an informative prior on Meta's alpha_max.
# Roughly: at full saturation, alpha_max represents the maximum incremental
# revenue contribution. Use the lift estimate and a multiplier to translate
# revenue per krona into total channel ceiling.
total_meta_spend_per_week = mmm["meta_S"].mean()
prior_alpha_max_meta_mean = lift_mean * total_meta_spend_per_week * 1.4
prior_alpha_max_meta_sd = lift_sd * total_meta_spend_per_week * 3.0
print(f"\nDerived prior on Meta alpha_max: "
      f"Normal({prior_alpha_max_meta_mean:,.0f}, {prior_alpha_max_meta_sd:,.0f})")

# -----------------------------------------------------------------------------
# Step 2: MMM with calibrated prior
# -----------------------------------------------------------------------------
paid_channels = ["search_S", "meta_S", "tiktok_S", "youtube_S", "display_S"]
channel_names = [c.replace("_S", "") for c in paid_channels]
META_IDX = channel_names.index("meta") + 1  # Stan is 1-indexed
N = len(mmm)
t_idx = np.arange(N, dtype=float)

mmm_data = {
    "N": N,
    "C": len(paid_channels),
    "X_paid": mmm[paid_channels].to_numpy(),
    "X_nl": mmm["newsletter"].to_numpy(),
    "competitor": mmm["competitor_sales"].to_numpy(),
    "events": mmm["events"].astype(int).tolist(),
    "t_idx": t_idx,
    "s_cos": np.cos(2 * np.pi * t_idx / 52),
    "s_sin": np.sin(2 * np.pi * t_idx / 52),
    "revenue": mmm["revenue"].to_numpy(),
    "meta_idx": META_IDX,
    "prior_alpha_max_meta_mean": float(prior_alpha_max_meta_mean),
    "prior_alpha_max_meta_sd": float(prior_alpha_max_meta_sd),
}

print("\nFitting calibrated MMM")
mmm_model = CmdStanModel(stan_file="stan/mmm_calibrated.stan")
mmm_fit = mmm_model.sample(data=mmm_data, chains=4, parallel_chains=4,
                            iter_warmup=1000, iter_sampling=1000, seed=42,
                            show_progress=True, adapt_delta=0.95,
                            max_treedepth=12)
print(mmm_fit.diagnose())

mmm_idata = az.from_cmdstanpy(mmm_fit)

# Also fit an uncalibrated baseline (same model with wide prior on Meta)
print("\nFitting uncalibrated MMM (baseline)")
mmm_data_uncal = {**mmm_data,
                   "prior_alpha_max_meta_mean": 30000.0,
                   "prior_alpha_max_meta_sd": 30000.0}
mmm_fit_uncal = mmm_model.sample(data=mmm_data_uncal, chains=4, parallel_chains=4,
                                  iter_warmup=1000, iter_sampling=1000, seed=42,
                                  show_progress=True, adapt_delta=0.95,
                                  max_treedepth=12)
mmm_idata_uncal = az.from_cmdstanpy(mmm_fit_uncal)

# -----------------------------------------------------------------------------
# Compare Meta posterior: calibrated vs uncalibrated
# -----------------------------------------------------------------------------
meta_cal = mmm_fit.stan_variable("alpha_max")[:, channel_names.index("meta")]
meta_uncal = mmm_fit_uncal.stan_variable("alpha_max")[:, channel_names.index("meta")]
true_meta = gt[gt["channel"] == "meta"]["alpha_max"].values[0]

print("\nMeta alpha_max posterior comparison:")
for label, samples in [("Uncalibrated", meta_uncal), ("Calibrated", meta_cal)]:
    mean = samples.mean()
    lo, hi = np.percentile(samples, [5, 95])
    width = hi - lo
    inside = "yes" if lo <= true_meta <= hi else "NO"
    print(f"  {label:<14} mean={mean:>8.0f}  90% CI [{lo:>7.0f}, {hi:>7.0f}]"
          f"  width={width:>7.0f}  contains truth: {inside}")
print(f"  Truth: {true_meta}")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(meta_uncal, bins=40, alpha=0.5, label="Uncalibrated", color="steelblue")
ax.hist(meta_cal,   bins=40, alpha=0.5, label="Calibrated",   color="darkorange")
ax.axvline(true_meta, color="black", linestyle="--", lw=1.5,
           label=f"Truth ({true_meta:,})")
ax.set_title("Meta alpha_max posterior: calibrated vs uncalibrated")
ax.set_xlabel("alpha_max (SEK)")
ax.set_ylabel("Posterior density")
ax.legend()
plt.tight_layout()
plt.savefig("figures/01_meta_calibration.png", dpi=120)
plt.close()

# Lift posterior
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(lift_samples, bins=40, color="seagreen", alpha=0.8)
true_lift = float(exp_truth.set_index("param").loc["incremental_roas_meta", "value"])
ax.axvline(true_lift, color="black", linestyle="--", lw=1.5,
           label=f"Truth ({true_lift:.2f})")
ax.set_title("Posterior of incremental Meta ROAS from geo-lift")
ax.set_xlabel("Incremental ROAS")
ax.set_ylabel("Posterior density")
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_lift_posterior.png", dpi=120)
plt.close()

print("\nFigures saved. Done.")
