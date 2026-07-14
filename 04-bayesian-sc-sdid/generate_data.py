"""Generate synthetic geo panel data with a known treatment effect.

Panel structure: N units (regions), T weeks. One treated unit gets an
additive lift for t > T0. The remaining N-1 units are controls.

Data-generating model:
    Y[i, t] = mu[i] + delta[t] + tau * 1[i = treated, t > T0] + eps[i, t]

- mu[i]: unit-specific baseline level
- delta[t]: shared time effect (trend + annual cosine seasonality)
- tau:     true ATT (per-week treatment effect)
- eps:     iid Gaussian noise

Panel size is deliberately larger than a minimal illustration (51 units,
80 weeks, 52 pre) so that the omega and lambda posteriors are well
identified and 90% credible intervals achieve nominal coverage.
"""
import numpy as np
import pandas as pd

# Use the modern Generator API so this file matches the Monte Carlo study
# (mc_study.py) that draws data via np.random.default_rng.
SEED = 19
rng = np.random.default_rng(SEED)

N = 51                # 1 treated + 50 control units
T = 80                # 80 weeks total
T0 = 52               # last pre-treatment week (52 pre + 28 post)
TREATED = 0           # index of treated unit
TRUE_TAU = 25_000     # true per-week treatment effect

# Unit-specific baselines
mu = rng.uniform(150_000, 350_000, size=N)

# Shared time effect
t = np.arange(T, dtype=float)
trend = 500 * t
season = 20_000 * np.cos(2 * np.pi * t / 52)
delta = trend + season

# Outcome matrix
Y = np.zeros((N, T))
for i in range(N):
    Y[i, :] = mu[i] + delta + rng.normal(0, 8_000, size=T)

# Treatment effect on the treated unit, post-period only
Y[TREATED, T0:] += TRUE_TAU

# Long-format DataFrame
rows = []
for i in range(N):
    for time in range(T):
        rows.append({
            "unit": i,
            "week": time,
            "treated": int(i == TREATED),
            "post": int(time >= T0),
            "Y": Y[i, time],
        })
panel = pd.DataFrame(rows)
panel.to_csv("data/panel.csv", index=False)

# Ground truth
gt = pd.DataFrame({
    "param": ["N", "T", "T0", "treated_unit", "true_tau", "noise_sigma"],
    "value": [N, T, T0, TREATED, TRUE_TAU, 8_000.0],
})
gt.to_csv("data/ground_truth.csv", index=False)

# Wide-format outcomes for fast Stan loading
wide_Y = pd.DataFrame(Y.T, columns=[f"unit_{i}" for i in range(N)])
wide_Y.insert(0, "week", np.arange(T))
wide_Y.to_csv("data/Y_wide.csv", index=False)

print(f"Saved data/panel.csv ({len(panel)} rows)")
print(f"N units: {N}, T weeks: {T}, T0: {T0}, treated unit: {TREATED}")
print(f"True ATT (tau): {TRUE_TAU:,} per week")
