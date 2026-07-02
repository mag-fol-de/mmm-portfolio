"""Fit the Bayesian MMM in Stan, run diagnostics, plot decomposition and ROAS.

Run with: uv run python fit_mmm.py

Requires cmdstan installed once:
    uv run python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
df = pd.read_csv("data/marketing_data.csv", parse_dates=["DATE"]).sort_values("DATE")
gt = pd.read_csv("data/ground_truth.csv")

paid_channels = ["search_S", "meta_S", "tiktok_S", "youtube_S", "display_S"]
channel_names = [c.replace("_S", "") for c in paid_channels]
N = len(df)
C = len(paid_channels)

t_idx = np.arange(N, dtype=float)
s_cos = np.cos(2 * np.pi * t_idx / 52)
s_sin = np.sin(2 * np.pi * t_idx / 52)

stan_data = {
    "N": N,
    "C": C,
    "X_paid": df[paid_channels].to_numpy(),
    "X_nl": df["newsletter"].to_numpy(),
    "competitor": df["competitor_sales"].to_numpy(),
    "events": df["events"].astype(int).tolist(),
    "t_idx": t_idx,
    "s_cos": s_cos,
    "s_sin": s_sin,
    "revenue": df["revenue"].to_numpy(),
}

# -----------------------------------------------------------------------------
# 2. Compile + sample
# -----------------------------------------------------------------------------
print("Compiling Stan model")
model = CmdStanModel(stan_file="stan/mmm.stan")

print("Sampling")
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
    show_progress=True,
    adapt_delta=0.95,
    max_treedepth=12,
)

print(fit.diagnose())

idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive=["revenue_pred"],
    observed_data={"revenue": stan_data["revenue"]},
)

Path("samples").mkdir(exist_ok=True)
idata.to_netcdf("samples/posterior.nc")

# -----------------------------------------------------------------------------
# 3. Diagnostics
# -----------------------------------------------------------------------------
summary = az.summary(
    idata,
    var_names=["alpha_max", "lambda", "K", "alpha_nl", "lambda_nl", "K_nl",
               "beta0", "beta_trend", "beta_cos", "beta_sin",
               "beta_comp", "beta_ev", "sigma"],
    round_to=3,
)
print(summary)

print(f"\nMax R-hat: {summary['r_hat'].max():.4f}")
print(f"Min ESS bulk: {summary['ess_bulk'].min():.0f}")

# -----------------------------------------------------------------------------
# 4. Calibration vs ground truth
# -----------------------------------------------------------------------------
post = fit.stan_variables()

print("\nCalibration table (posterior mean and 90% CI vs ground truth):")
print(f"{'Channel':<10}{'Param':<12}{'True':>10}{'Mean':>10}{'5%':>10}{'95%':>10}")
print("-" * 62)

for i, ch in enumerate(channel_names):
    gt_row = gt[gt["channel"] == ch].iloc[0]
    for param_key, gt_key in [("alpha_max", "alpha_max"),
                               ("lambda", "lambda"),
                               ("K", "K")]:
        samples = post[param_key][:, i]
        mean = samples.mean()
        lo, hi = np.percentile(samples, [5, 95])
        true_val = gt_row[gt_key]
        flag = " " if lo <= true_val <= hi else "*"
        print(f"{ch:<10}{param_key:<12}{true_val:>10.3f}{mean:>10.3f}"
              f"{lo:>10.3f}{hi:>10.3f}  {flag}")

# -----------------------------------------------------------------------------
# 5. Posterior predictive plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["DATE"], df["revenue"], color="steelblue", label="Actual", lw=1.8)

mu_post = post["mu"]
mu_mean = mu_post.mean(axis=0)
mu_lo, mu_hi = np.percentile(mu_post, [5, 95], axis=0)

ax.plot(df["DATE"], mu_mean, color="darkorange", label="Posterior mean", lw=1.6)
ax.fill_between(df["DATE"], mu_lo, mu_hi, alpha=0.25, color="darkorange",
                label="90% CI")
ax.set_title("Actual vs modelled revenue")
ax.set_ylabel("Revenue (SEK)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/01_actual_vs_modeled.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# 6. Decomposition (posterior mean of per-driver contribution)
# -----------------------------------------------------------------------------
contrib_post = post["contribution"]  # shape (draws, N, C)
contrib_mean = contrib_post.mean(axis=0)  # (N, C)

decomp = pd.DataFrame(contrib_mean, columns=channel_names, index=df["DATE"])
decomp["Baseline"] = post["beta0"].mean()
decomp["Trend"]    = post["beta_trend"].mean() * t_idx
decomp["Season"]   = (post["beta_cos"].mean() * s_cos
                     + post["beta_sin"].mean() * s_sin)
decomp["Competitor"] = post["beta_comp"].mean() * df["competitor_sales"].to_numpy()
decomp["Events"]    = post["beta_ev"].mean() * df["events"].to_numpy()
decomp["Newsletter"] = (post["alpha_nl"].mean()
                        * post["nl_sat"].mean(axis=0))

fig, ax = plt.subplots(figsize=(13, 6))
plot_order = ["Baseline", "Trend", "Season", "Competitor",
              "search", "meta", "tiktok", "youtube", "display",
              "Newsletter", "Events"]
paid_set = {"search", "meta", "tiktok", "youtube", "display"}
for col in plot_order:
    lw = 2.0 if col in paid_set else 1.2
    alpha = 0.95 if col in paid_set else 0.7
    ax.plot(decomp.index, decomp[col], label=col, linewidth=lw, alpha=alpha)
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Revenue decomposition over time (posterior mean per driver)")
ax.set_ylabel("Revenue contribution (SEK)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
plt.tight_layout()
plt.savefig("figures/02_decomposition.png", dpi=120)
plt.close()

# Total contribution per driver (3 years)
totals = decomp.sum().sort_values(ascending=False)
total_rev = df["revenue"].sum()
attribution = pd.DataFrame({
    "contribution": totals,
    "share_of_revenue_pct": (totals / total_rev * 100).round(1),
})
print("\nAttribution over the full period:")
print(attribution)

# -----------------------------------------------------------------------------
# 7. ROAS with credible intervals
# -----------------------------------------------------------------------------
print("\nROAS per paid channel:")
print(f"{'Channel':<10}{'Spend':>14}{'Contrib mean':>16}"
      f"{'ROAS mean':>12}{'ROAS 5%':>10}{'ROAS 95%':>10}")
print("-" * 72)

roas_rows = []
for i, ch in enumerate(channel_names):
    spend_total = df[paid_channels[i]].sum()
    contrib_total_draws = contrib_post[:, :, i].sum(axis=1)
    roas_draws = contrib_total_draws / spend_total
    roas_mean = roas_draws.mean()
    roas_lo, roas_hi = np.percentile(roas_draws, [5, 95])
    contrib_mean_total = contrib_total_draws.mean()
    print(f"{ch:<10}{spend_total:>14,.0f}{contrib_mean_total:>16,.0f}"
          f"{roas_mean:>12.3f}{roas_lo:>10.3f}{roas_hi:>10.3f}")
    roas_rows.append({
        "channel": ch,
        "spend": spend_total,
        "contribution_mean": contrib_mean_total,
        "roas_mean": roas_mean,
        "roas_low": roas_lo,
        "roas_high": roas_hi,
    })

roas_df = pd.DataFrame(roas_rows).sort_values("roas_mean", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(roas_df["roas_mean"], roas_df["channel"],
            xerr=[roas_df["roas_mean"] - roas_df["roas_low"],
                  roas_df["roas_high"] - roas_df["roas_mean"]],
            fmt="o", color="darkgreen", capsize=4)
ax.axvline(1.0, color="black", linestyle="--", lw=0.8, label="ROAS = 1")
ax.set_title("ROAS by paid channel (posterior mean, 90% CI)")
ax.set_xlabel("Revenue per krona spent")
ax.legend()
plt.tight_layout()
plt.savefig("figures/03_roas.png", dpi=120)
plt.close()

# -----------------------------------------------------------------------------
# 8. Save outputs
# -----------------------------------------------------------------------------
summary.to_csv("samples/posterior_summary.csv")
attribution.to_csv("samples/attribution.csv")
roas_df.to_csv("samples/roas.csv", index=False)

print("\nDone. Outputs in samples/ and figures/.")
