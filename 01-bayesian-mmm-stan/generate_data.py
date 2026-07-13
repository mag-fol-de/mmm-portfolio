"""Synthetic weekly marketing panel for the Bayesian MMM.

Generates 156 weeks (3 years) of spend, revenue, controls and events.
Ground-truth channel parameters are chosen so realised ROAS lands in a
realistic industry range: Search 3-4x, Meta / TikTok 2-3x, YouTube
1-2x, Display around break-even. Effect sizes are calibrated to spend
levels rather than kept at nominal defaults.

Outputs
-------
data/marketing_data.csv : one row per week with revenue and covariates
data/ground_truth.csv   : per-channel (lambda, K, alpha_max) that the
                          MMM should recover.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260101)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

N = 156                                     # weeks (3 years)
t = np.arange(N)


# -----------------------------------------------------------------------------
# Weekly spend per paid channel (SEK). Always-on with seasonal amplification
# and holiday spikes so channels are correlated but distinct.
# -----------------------------------------------------------------------------
def build_spend(base, seasonal_amp, spike_weeks, spike_mult, sigma):
    trend = 1.0 + 0.15 * (t / N)              # slow growth
    season = 1.0 + seasonal_amp * np.sin(2 * np.pi * t / 52 - 1.0)
    noise = RNG.lognormal(mean=0.0, sigma=sigma, size=N)
    spikes = np.ones(N)
    for wk in spike_weeks:
        spikes[wk:wk + 2] *= spike_mult
    return base * trend * season * spikes * noise

search_S  = build_spend(22_000, 0.20, [45, 97, 149], 1.35, 0.15)
meta_S    = build_spend(50_000, 0.25, [45, 97, 149], 1.40, 0.20)
tiktok_S  = build_spend(32_000, 0.30, [45, 97, 149], 1.30, 0.22)
youtube_S = build_spend(20_000, 0.20, [45, 97, 149], 1.25, 0.18)
display_S = build_spend(14_500, 0.10, [45, 97, 149], 1.15, 0.15)

paid = {
    "search_S":  search_S,
    "meta_S":    meta_S,
    "tiktok_S":  tiktok_S,
    "youtube_S": youtube_S,
    "display_S": display_S,
}


# -----------------------------------------------------------------------------
# Ground-truth channel parameters. alpha_max chosen so that at typical
# post-adstock, post-saturation contribution, weekly channel contribution is
# a solid multiple of weekly spend for stronger channels.
# -----------------------------------------------------------------------------
gt = {
    "search":  {"lambda": 0.20, "K": 15_000,  "alpha_max": 150_000},  # ROAS ~ 4.0
    "meta":    {"lambda": 0.35, "K": 60_000,  "alpha_max": 220_000},  # ROAS ~ 2.5
    "tiktok":  {"lambda": 0.25, "K": 40_000,  "alpha_max": 150_000},  # ROAS ~ 2.4
    "youtube": {"lambda": 0.45, "K": 30_000,  "alpha_max": 65_000},   # ROAS ~ 1.7
    "display": {"lambda": 0.20, "K": 20_000,  "alpha_max": 32_000},   # ROAS ~ 1.2
}


def adstock(x, lam):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + lam * y[i - 1]
    return y


def hill(x, K):
    return x / (K + x)


channel_contribs = {}
for ch, spend in paid.items():
    key = ch.replace("_S", "")
    p = gt[key]
    ad = adstock(spend, p["lambda"])
    sat = hill(ad, p["K"])
    channel_contribs[key] = p["alpha_max"] * sat


# -----------------------------------------------------------------------------
# Newsletter (owned channel)
# -----------------------------------------------------------------------------
newsletter = RNG.poisson(lam=0.6, size=N)
newsletter = newsletter * RNG.uniform(3000, 6000, size=N)      # revenue-per-send times sends
nl_ad = adstock(newsletter, 0.15)
nl_sat = hill(nl_ad, 5000)
alpha_nl_true = 8_000
nl_contrib = alpha_nl_true * nl_sat


# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
competitor_sales = RNG.normal(loc=280_000, scale=45_000, size=N)
events = np.zeros(N, dtype=int)
events[[45, 46, 97, 98, 149, 150]] = 1                              # Black Friday / Christmas
s_cos = np.cos(2 * np.pi * t / 52)
s_sin = np.sin(2 * np.pi * t / 52)


# -----------------------------------------------------------------------------
# Baseline + assembly
# -----------------------------------------------------------------------------
beta0_true  = 260_000
beta_trend  = 400
beta_cos    = 45_000
beta_sin    = 15_000
beta_comp   = 0.35
beta_ev     = 55_000
sigma_true  = 24_000

mu = (
    beta0_true
    + beta_trend * t
    + beta_cos * s_cos
    + beta_sin * s_sin
    + beta_comp * competitor_sales
    + beta_ev * events
    + nl_contrib
    + sum(channel_contribs.values())
)

revenue = mu + RNG.normal(0, sigma_true, size=N)


# -----------------------------------------------------------------------------
# Assemble and write
# -----------------------------------------------------------------------------
dates = pd.date_range("2023-01-02", periods=N, freq="W-MON")
df = pd.DataFrame({
    "DATE": dates,
    "revenue": np.round(revenue, 2),
    **{k: np.round(v, 2) for k, v in paid.items()},
    "competitor_sales": np.round(competitor_sales, 2),
    "events": events,
    "newsletter": np.round(newsletter, 2),
})
df.to_csv(DATA / "marketing_data.csv", index=False)

pd.DataFrame([
    {"channel": ch, **params} for ch, params in gt.items()
]).to_csv(DATA / "ground_truth.csv", index=False)


# -----------------------------------------------------------------------------
# Sanity print: expected ROAS per channel over the panel
# -----------------------------------------------------------------------------
print(f"Generated {N} weeks of data.")
print(f"Mean weekly revenue: {revenue.mean():,.0f} SEK")
print(f"Total revenue: {revenue.sum()/1e6:.1f}m SEK")
print("\nExpected ROAS per paid channel (contribution / spend):")
for key in gt:
    contrib_total = float(channel_contribs[key].sum())
    spend_total   = float(paid[f"{key}_S"].sum())
    print(f"  {key:<10s} spend={spend_total/1e6:.2f}m  "
          f"contrib={contrib_total/1e6:.2f}m  ROAS={contrib_total/spend_total:.2f}")
