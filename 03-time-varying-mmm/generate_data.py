"""Generate synthetic weekly MMM data where one channel's max effect drifts
over time.

The drift is built into Search's alpha_max: it grows from 18,000 in the first
week to 38,000 by the last week. This mimics, for example, search demand
expanding as the customer base grows, or a privacy event (iOS 14) gradually
shifting attribution between channels.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

start = pd.Timestamp("2022-01-03")
n = 156
dates = pd.date_range(start=start, periods=n, freq="W-MON")
t = np.arange(n)

baseline = 250_000
trend = 250 * t
season = 55_000 * np.cos(2 * np.pi * (t - 50) / 52)
season_q = 18_000 * np.sin(2 * np.pi * t / 13)

def burst(low, high, p_active, size, smooth=2):
    active = np.random.binomial(1, p_active, size=size)
    spend = active * np.random.uniform(low, high, size=size)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        spend = np.convolve(spend, kernel, mode="same")
    return spend

search_S   = np.random.uniform(10_000, 35_000, size=n)
meta_S     = np.random.uniform(20_000, 80_000, size=n)
tiktok_S   = np.random.uniform(15_000, 50_000, size=n)
youtube_S  = burst(10_000, 70_000, 0.55, n, smooth=2)
display_S  = np.random.uniform(5_000, 25_000, size=n)

def adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + decay * out[i - 1]
    return out

def saturate(x, K):
    return x / (K + x)

LAMBDA = {"search": 0.20, "meta": 0.35, "tiktok": 0.25,
          "youtube": 0.45, "display": 0.20}
K = {"search": 15_000, "meta": 60_000, "tiktok": 40_000,
     "youtube": 30_000, "display": 20_000}

# Time-varying alpha_max for Search: linear ramp from 90k to 200k.
# Other channels: fixed at realistic industry ROAS levels.
alpha_max_search_t = np.linspace(90_000, 200_000, n)
ALPHA_FIXED = {"meta": 220_000, "tiktok": 150_000, "youtube": 65_000, "display": 32_000}

search_ad  = adstock(search_S,  LAMBDA["search"])
meta_ad    = adstock(meta_S,    LAMBDA["meta"])
tiktok_ad  = adstock(tiktok_S,  LAMBDA["tiktok"])
youtube_ad = adstock(youtube_S, LAMBDA["youtube"])
display_ad = adstock(display_S, LAMBDA["display"])

search_eff  = alpha_max_search_t * saturate(search_ad,  K["search"])
meta_eff    = ALPHA_FIXED["meta"]    * saturate(meta_ad,    K["meta"])
tiktok_eff  = ALPHA_FIXED["tiktok"]  * saturate(tiktok_ad,  K["tiktok"])
youtube_eff = ALPHA_FIXED["youtube"] * saturate(youtube_ad, K["youtube"])
display_eff = ALPHA_FIXED["display"] * saturate(display_ad, K["display"])

competitor_sales = (100_000
                    + 20_000 * np.cos(2 * np.pi * (t - 30) / 52)
                    + np.random.normal(0, 8_000, size=n))
competitor_effect = -0.05 * competitor_sales

events = np.zeros(n, dtype=int)
weeks_of_year = pd.Series(dates).dt.isocalendar().week.values
events[(weeks_of_year == 47) | (weeks_of_year == 51)] = 1
events_effect = 40_000 * events

newsletter = burst(2_000, 8_000, 0.7, n, smooth=2)
newsletter_effect = 6_000 * saturate(adstock(newsletter, 0.25), 5_000)

noise = np.random.normal(0, 14_000, size=n)
revenue = (baseline + trend + season + season_q
           + search_eff + meta_eff + tiktok_eff + youtube_eff + display_eff
           + competitor_effect + events_effect + newsletter_effect + noise)
revenue = np.maximum(revenue, 0)

df = pd.DataFrame({
    "DATE": dates.strftime("%Y-%m-%d"),
    "revenue": revenue.round(2),
    "search_S": search_S.round(2),
    "meta_S": meta_S.round(2),
    "tiktok_S": tiktok_S.round(2),
    "youtube_S": youtube_S.round(2),
    "display_S": display_S.round(2),
    "competitor_sales": competitor_sales.round(2),
    "events": events,
    "newsletter": newsletter.round(2),
})
df.to_csv("data/marketing_data.csv", index=False)

# Ground truth: search has time-varying alpha_max, others are fixed
gt_static = pd.DataFrame({
    "channel": ["meta", "tiktok", "youtube", "display"],
    "lambda":   [LAMBDA[c] for c in ["meta", "tiktok", "youtube", "display"]],
    "K":        [K[c]      for c in ["meta", "tiktok", "youtube", "display"]],
    "alpha_max": [ALPHA_FIXED[c] for c in ["meta", "tiktok", "youtube", "display"]],
})
gt_static.to_csv("data/ground_truth_static.csv", index=False)

gt_search = pd.DataFrame({
    "week": t,
    "DATE": dates.strftime("%Y-%m-%d"),
    "alpha_max_search_true": alpha_max_search_t,
})
gt_search.to_csv("data/ground_truth_search_alpha_t.csv", index=False)

print(f"Saved marketing_data.csv ({len(df)} rows)")
print(f"Saved ground_truth_static.csv (4 fixed channels)")
print(f"Saved ground_truth_search_alpha_t.csv (search alpha_max over time)")
print()
print(f"Search alpha_max ramp: {alpha_max_search_t[0]:.0f} -> {alpha_max_search_t[-1]:.0f}")
print(df.head())
