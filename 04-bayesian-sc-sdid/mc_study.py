"""Monte Carlo study: fit BSC and BSDID on many simulated panels, then
report average coverage, bias, CI width, and absolute error per model.

Panel structure is fixed (N=51, T=80, T0=52). Only the random seed varies
between runs. For each seed we regenerate the panel, fit both Stan
models, and record posterior summary statistics.

Outputs
-------
samples/mc_results.csv    : per-seed metrics for both models
samples/mc_summary.csv    : aggregated metrics (coverage rate, mean bias,
                            mean CI width, mean |bias|) per model
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

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "samples"
OUT.mkdir(exist_ok=True)

# Panel size (matches generate_data.py)
N = 51
T = 80
T0 = 52
TREATED = 0
TRUE_TAU = 25_000

# Monte Carlo settings
N_SEEDS = 20
SEEDS = list(range(1, N_SEEDS + 1))
STAN_SEED = 4242  # fixed sampler seed so only data varies across MC iterations


def simulate_panel(seed: int) -> np.ndarray:
    """Generate one panel (N x T) with the fixed data-generating process."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(150_000, 350_000, size=N)
    t = np.arange(T, dtype=float)
    delta = 500 * t + 20_000 * np.cos(2 * np.pi * t / 52)
    Y = np.zeros((N, T))
    for i in range(N):
        Y[i, :] = mu[i] + delta + rng.normal(0, 8_000, size=T)
    Y[TREATED, T0:] += TRUE_TAU
    return Y


def fit_and_summarize(model, Y):
    """Fit a model and return ATT mean, 90% CI, coverage flag."""
    Y_treated = Y[TREATED]
    Y_control = np.delete(Y, TREATED, axis=0).T          # (T, N-1)
    data = {
        "T": T, "T0": T0, "N": Y_control.shape[1],
        "Y_treated": Y_treated,
        "Y_control": Y_control,
    }
    fit = model.sample(
        data=data, chains=4, parallel_chains=4,
        iter_warmup=1000, iter_sampling=1000, seed=STAN_SEED,
        show_progress=False,
    )
    att = fit.stan_variable("att_mean")
    mean = float(att.mean())
    lo, hi = np.percentile(att, [5, 95])
    covers = bool(lo <= TRUE_TAU <= hi)
    return {
        "mean": mean,
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "bias": mean - TRUE_TAU,
        "abs_bias": abs(mean - TRUE_TAU),
        "covers": covers,
    }


def main():
    print(f"Compiling Stan models...")
    sc_model   = CmdStanModel(stan_file=str(ROOT / "stan" / "bayesian_sc.stan"))
    sdid_model = CmdStanModel(stan_file=str(ROOT / "stan" / "bayesian_sdid.stan"))

    records = []
    for k, seed in enumerate(SEEDS, start=1):
        print(f"[{k:2d}/{len(SEEDS)}] seed={seed}", end=" ", flush=True)
        Y = simulate_panel(seed)
        sc = fit_and_summarize(sc_model, Y)
        sdid = fit_and_summarize(sdid_model, Y)
        print(f"| SC mean={sc['mean']:>8,.0f} covers={sc['covers']!s:>5} "
              f"| SDID mean={sdid['mean']:>8,.0f} covers={sdid['covers']!s:>5}")
        records.append({
            "seed": seed,
            "sc_mean": sc["mean"], "sc_lo": sc["lo"], "sc_hi": sc["hi"],
            "sc_width": sc["width"], "sc_bias": sc["bias"], "sc_covers": sc["covers"],
            "sdid_mean": sdid["mean"], "sdid_lo": sdid["lo"], "sdid_hi": sdid["hi"],
            "sdid_width": sdid["width"], "sdid_bias": sdid["bias"], "sdid_covers": sdid["covers"],
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT / "mc_results.csv", index=False)

    summary = pd.DataFrame({
        "model":       ["BSC", "BSDID"],
        "coverage_pct": [df["sc_covers"].mean() * 100, df["sdid_covers"].mean() * 100],
        "mean_bias":    [df["sc_bias"].mean(),         df["sdid_bias"].mean()],
        "mean_abs_bias":[df["sc_bias"].abs().mean(),   df["sdid_bias"].abs().mean()],
        "mean_ci_width":[df["sc_width"].mean(),        df["sdid_width"].mean()],
    })
    summary.to_csv(OUT / "mc_summary.csv", index=False)

    print("\n== Monte Carlo summary ==")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
