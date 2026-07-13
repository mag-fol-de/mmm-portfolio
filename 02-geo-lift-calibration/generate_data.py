"""Generate two datasets:

1. Geo-lift experiment data: two regions, four weeks of pre-period and four
   weeks of experiment period. Treatment region gets a known boost in Meta
   spend during the experiment.

2. Standard MMM weekly dataset (same structure as project 1) for the full
   period. Reuses project 1's data-generating process but stores ground
   truth separately.
"""
import numpy as np
import pandas as pd

np.random.seed(7)

# -----------------------------------------------------------------------------
# Geo-lift experiment
# -----------------------------------------------------------------------------
n_pre = 8       # 8 weeks pre-period
n_exp = 8       # 8 weeks experiment
n_total = n_pre + n_exp

# Baseline weekly revenue per region
mu_A = 200_000
mu_B = 180_000

# Meta spend pattern: both regions have always-on spend during pre.
# During experiment, region B gets +100k extra in Meta.
np.random.seed(11)
meta_base_A = np.random.uniform(20_000, 60_000, size=n_total)
meta_base_B = np.random.uniform(20_000, 60_000, size=n_total)
meta_boost_B = np.zeros(n_total)
meta_boost_B[n_pre:] = 100_000   # treatment for the second half

# True lift per krona of incremental Meta spend in the experiment.
# This is what the geo-lift Bayesian model should recover.
TRUE_INCREMENTAL_ROAS_META = 0.55

# Generate revenue with noise
noise_A = np.random.normal(0, 6_000, size=n_total)
noise_B = np.random.normal(0, 6_000, size=n_total)

rev_A = mu_A + 0.30 * meta_base_A + noise_A
rev_B = (mu_B
         + 0.30 * meta_base_B
         + TRUE_INCREMENTAL_ROAS_META * meta_boost_B
         + noise_B)

geo = pd.DataFrame({
    "week":        list(range(n_total)) * 2,
    "region":      ["A"] * n_total + ["B"] * n_total,
    "is_treated":  [0] * n_total + [0] * n_pre + [1] * n_exp,
    "meta_spend":  np.concatenate([meta_base_A, meta_base_B + meta_boost_B]),
    "meta_boost":  np.concatenate([np.zeros(n_total), meta_boost_B]),
    "revenue":     np.concatenate([rev_A, rev_B]),
})
geo.to_csv("data/geo_experiment.csv", index=False)

# -----------------------------------------------------------------------------
# MMM dataset (same DGP as project 1)
# -----------------------------------------------------------------------------
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
ALPHA_MAX = {"search": 150_000, "meta": 220_000, "tiktok": 150_000,
             "youtube": 65_000, "display": 32_000}

def channel_effect(spend, name):
    return ALPHA_MAX[name] * saturate(adstock(spend, LAMBDA[name]), K[name])

search_eff  = channel_effect(search_S,  "search")
meta_eff    = channel_effect(meta_S,    "meta")
tiktok_eff  = channel_effect(tiktok_S,  "tiktok")
youtube_eff = channel_effect(youtube_S, "youtube")
display_eff = channel_effect(display_S, "display")

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

gt = pd.DataFrame({
    "channel": list(LAMBDA.keys()),
    "lambda": list(LAMBDA.values()),
    "K": list(K.values()),
    "alpha_max": list(ALPHA_MAX.values()),
})
gt.to_csv("data/ground_truth.csv", index=False)

# Save the experiment ground truth too
exp_gt = pd.DataFrame({
    "param": ["incremental_roas_meta", "true_meta_alpha_max"],
    "value": [TRUE_INCREMENTAL_ROAS_META, ALPHA_MAX["meta"]],
})
exp_gt.to_csv("data/experiment_truth.csv", index=False)

print("Saved data/geo_experiment.csv ({} rows)".format(len(geo)))
print("Saved data/marketing_data.csv ({} rows)".format(len(df)))
print("Saved data/ground_truth.csv and data/experiment_truth.csv")
print()
print("Experiment summary:")
print(geo.groupby(["region", "is_treated"]).agg(
    weeks=("revenue", "count"),
    mean_rev=("revenue", "mean"),
    mean_meta=("meta_spend", "mean"),
).round(0))
