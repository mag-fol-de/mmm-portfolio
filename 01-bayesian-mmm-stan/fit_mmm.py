"""Fit the Bayesian MMM in Stan, run diagnostics, plot decomposition and ROAS.

Sets CMDSTAN and PATH for local cmdstan + RTools40 toolchain.
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
# 1. Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "marketing_data.csv",
                 parse_dates=["DATE"]).sort_values("DATE").reset_index(drop=True)
gt = pd.read_csv(ROOT / "data" / "ground_truth.csv")

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
    "t_idx": t_idx.tolist(),
    "s_cos": s_cos.tolist(),
    "s_sin": s_sin.tolist(),
    "revenue": df["revenue"].to_numpy(),
}

# -----------------------------------------------------------------------------
# 2. Compile + sample
# -----------------------------------------------------------------------------
print("Compiling Stan model...")
model = CmdStanModel(stan_file=str(ROOT / "stan" / "mmm.stan"))

print("Sampling...")
fit = model.sample(
    data=stan_data, chains=4, parallel_chains=4,
    iter_warmup=1000, iter_sampling=1000, seed=42,
    show_progress=False, adapt_delta=0.95, max_treedepth=12,
)
print(fit.diagnose())

post = fit.stan_variables()

# -----------------------------------------------------------------------------
# 3. Parameter recovery vs ground truth
# -----------------------------------------------------------------------------
print("\nParameter recovery (posterior mean, 90% CI, ground truth):")
print(f"{'Channel':<10}{'Param':<12}{'True':>10}{'Mean':>10}{'5%':>10}{'95%':>10}  Covers")
print("-" * 72)

recovery_rows = []
for i, ch in enumerate(channel_names):
    gt_row = gt[gt["channel"] == ch].iloc[0]
    for param_key in ["alpha_max", "lambda", "K"]:
        samples = post[param_key][:, i]
        mean = samples.mean()
        lo, hi = np.percentile(samples, [5, 95])
        true_val = float(gt_row[param_key])
        covers = lo <= true_val <= hi
        flag = "yes" if covers else "NO"
        print(f"{ch:<10}{param_key:<12}{true_val:>10.3f}{mean:>10.3f}"
              f"{lo:>10.3f}{hi:>10.3f}  {flag}")
        recovery_rows.append(dict(
            channel=ch, param=param_key, true=true_val,
            mean=mean, p05=lo, p95=hi, covers=covers,
        ))

pd.DataFrame(recovery_rows).to_csv(OUT / "parameter_recovery.csv", index=False)

# -----------------------------------------------------------------------------
# 4. Model fit
# -----------------------------------------------------------------------------
mu_post = post["mu"]
mu_mean = mu_post.mean(axis=0)
resid = df["revenue"].to_numpy() - mu_mean
ss_res = float(np.sum(resid ** 2))
ss_tot = float(np.sum((df["revenue"] - df["revenue"].mean()) ** 2))
r2 = 1 - ss_res / ss_tot
rmse = float(np.sqrt(np.mean(resid ** 2)))
mae = float(np.mean(np.abs(resid)))
mape = float(np.mean(np.abs(resid) / df["revenue"].to_numpy()))

print(f"\nModel fit:")
print(f"  R^2:  {r2:.4f}")
print(f"  RMSE: {rmse:,.0f}")
print(f"  MAE:  {mae:,.0f}")
print(f"  MAPE: {mape:.3%}")

# -----------------------------------------------------------------------------
# 5. Actual vs modelled
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120})
fig, ax = plt.subplots(figsize=(12, 5))
# Posterior predictive interval on revenue: uses revenue_pred = Normal(mu, sigma)
# so the band captures BOTH parameter uncertainty AND observation noise, not
# only the credible interval on the mean.
revenue_pred_post = post["revenue_pred"]                       # (draws, N)
pred_lo, pred_hi = np.percentile(revenue_pred_post, [5, 95], axis=0)
coverage = float(((df["revenue"].to_numpy() >= pred_lo) & (df["revenue"].to_numpy() <= pred_hi)).mean())
ax.plot(df["DATE"], df["revenue"], color="steelblue", label="Actual", lw=1.8)
ax.plot(df["DATE"], mu_mean, color="darkorange", label="Posterior mean", lw=1.4)
ax.fill_between(df["DATE"], pred_lo, pred_hi, alpha=0.25, color="darkorange",
                label=f"90% posterior predictive interval ({coverage:.0%} coverage)")
ax.set_title(f"Actual vs modelled revenue (R^2 = {r2:.3f})")
ax.set_ylabel("Revenue (SEK)")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "01_actual_vs_modeled.png"); plt.close()

# -----------------------------------------------------------------------------
# 6. Revenue decomposition
# -----------------------------------------------------------------------------
contrib_post = post["contribution"]              # (draws, N, C)
contrib_mean = contrib_post.mean(axis=0)         # (N, C)

decomp = pd.DataFrame(contrib_mean, columns=channel_names, index=df["DATE"])
decomp["Baseline"] = post["beta0"].mean()
decomp["Trend"]    = post["beta_trend"].mean() * t_idx
decomp["Season"]   = post["beta_cos"].mean() * s_cos + post["beta_sin"].mean() * s_sin
decomp["Competitor"] = post["beta_comp"].mean() * df["competitor_sales"].to_numpy()
decomp["Events"]     = post["beta_ev"].mean() * df["events"].to_numpy()
decomp["Newsletter"] = post["alpha_nl"].mean() * post["nl_sat"].mean(axis=0)

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
plt.tight_layout(); plt.savefig(FIG / "02_decomposition.png"); plt.close()

# Attribution totals
totals = decomp.sum().sort_values(ascending=False)
total_rev = df["revenue"].sum()
attribution = pd.DataFrame({
    "driver": totals.index,
    "contribution": totals.values,
    "share_pct": (totals.values / total_rev * 100).round(2),
})
attribution.to_csv(OUT / "attribution.csv", index=False)
print("\nAttribution (posterior mean):")
print(attribution.to_string(index=False))

# -----------------------------------------------------------------------------
# 7. ROAS per paid channel with 90% CI
# -----------------------------------------------------------------------------
print("\nROAS per paid channel:")
print(f"{'Channel':<10}{'Spend':>14}{'Contribution':>16}{'ROAS mean':>12}{'ROAS 5%':>10}{'ROAS 95%':>10}")
print("-" * 72)

roas_rows = []
for i, ch in enumerate(channel_names):
    spend_total = float(df[paid_channels[i]].sum())
    contrib_total_draws = contrib_post[:, :, i].sum(axis=1)     # (draws,)
    roas_draws = contrib_total_draws / spend_total
    roas_mean = float(roas_draws.mean())
    roas_lo, roas_hi = np.percentile(roas_draws, [5, 95])
    contrib_mean_total = float(contrib_total_draws.mean())
    print(f"{ch:<10}{spend_total:>14,.0f}{contrib_mean_total:>16,.0f}"
          f"{roas_mean:>12.3f}{roas_lo:>10.3f}{roas_hi:>10.3f}")
    roas_rows.append(dict(
        channel=ch, spend=spend_total, contribution=contrib_mean_total,
        roas_mean=roas_mean, roas_low=float(roas_lo), roas_high=float(roas_hi),
    ))

roas_df = pd.DataFrame(roas_rows).sort_values("roas_mean")
roas_df.to_csv(OUT / "roas.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(roas_df["roas_mean"], roas_df["channel"],
            xerr=[roas_df["roas_mean"] - roas_df["roas_low"],
                  roas_df["roas_high"] - roas_df["roas_mean"]],
            fmt="o", color="darkgreen", capsize=4, markersize=8)
ax.axvline(1.0, color="black", linestyle="--", lw=0.8, label="ROAS = 1")
ax.set_title("ROAS by paid channel (posterior mean, 90% CI)")
ax.set_xlabel("Revenue per krona spent")
ax.legend()
plt.tight_layout(); plt.savefig(FIG / "03_roas.png"); plt.close()

# -----------------------------------------------------------------------------
# 8. Metrics summary
# -----------------------------------------------------------------------------
pd.Series({
    "R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape,
    "n_weeks": N, "n_channels": C,
    "coverage_alpha_max": sum(1 for r in recovery_rows
                              if r["param"] == "alpha_max" and r["covers"]),
    "coverage_lambda":    sum(1 for r in recovery_rows
                              if r["param"] == "lambda" and r["covers"]),
    "coverage_K":         sum(1 for r in recovery_rows
                              if r["param"] == "K" and r["covers"]),
}).to_csv(OUT / "metrics.csv")

print(f"\nFigures written to {FIG}")
print(f"Tables written to {OUT}")
print("Done.")
