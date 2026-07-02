# Geo-Lift Calibration

Second project in the MMM portfolio.

## Goal

Run a Bayesian geo-lift experiment estimation, translate its posterior into
an informative prior on Meta's `alpha_max` in the MMM, and show that the
calibrated MMM produces a tighter and more accurate posterior than the
uncalibrated baseline.

This mirrors the reference stated calibration practice: "models are
calibrated with the best available information for each media (lift tests,
attribution data, the reference benchmarks)".

## Why this matters

A vanilla MMM relies entirely on observational data. Channel coefficients
are weakly identified because of multicollinearity (channels move together
during Q4 campaigns, post-Christmas slumps, etc.).

A lift test isolates a single channel's causal effect by holding all other
factors constant across regions. The resulting estimate is *experimentally
identified*, not just correlationally. Feeding that estimate back into the
MMM as a prior anchors the channel coefficient and resolves the
identification problem.

## Pipeline

```
geo_experiment.csv
   |
   v
Stan model 1 (lift.stan)
estimate posterior of incremental ROAS on Meta
   |
   v
Translate to a Normal prior on Meta's alpha_max
   |
   v
Stan model 2 (mmm_calibrated.stan)
full MMM with the informative prior on Meta
   |
   v
Compare posterior to an uncalibrated MMM
```

## Data

- `data/geo_experiment.csv`: 2 regions x 16 weeks. First 8 weeks pre-period,
  last 8 weeks experiment. Region B gets +100k extra Meta spend during the
  experiment.
- `data/marketing_data.csv`: same 156-week MMM data as project 1.
- `data/experiment_truth.csv`: the true incremental ROAS (0.55) that the
  geo-lift model should recover.
- `data/ground_truth.csv`: the true MMM parameters per channel.

## How to run

```bash
uv run python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
uv run python generate_data.py    # regenerates data if needed
uv run python fit_calibration.py  # runs both Stan models, produces plots
```

## Outputs

| File | Content |
|---|---|
| `figures/01_meta_calibration.png` | Meta posterior: calibrated (orange) vs uncalibrated (blue) vs truth (dashed) |
| `figures/02_lift_posterior.png` | Posterior of the incremental ROAS from the geo-lift |

## Expected result

The calibrated posterior should be **narrower** (lower posterior variance)
and **closer to the truth** than the uncalibrated baseline. This is the
empirical proof that calibration tightens MMM estimates.

## Modelling decisions worth defending

- **Difference-in-differences via Stan, not a t-test**: gives a full
  posterior on the lift rather than a point estimate plus p-value. Lets you
  propagate uncertainty into the MMM prior.
- **Region intercepts**: each region has its own baseline. Without them
  the model would conflate region differences with treatment effect.
- **Lift parameter constrained positive**: a true treatment effect should
  increase revenue, not decrease it. The constraint reflects prior belief
  and avoids weird modes from short experiments.
- **Translating posterior into MMM prior**: the simplest version uses
  the posterior mean and sd, scaled by typical Meta spend to convert
  per-krona ROAS into a channel-ceiling parameter. Worth noting in the
  report: this is approximate and a fully consistent treatment would use a
  single hierarchical model.

## Limitations

- Synthetic data with linear lift model. Real lift tests have non-linear
  saturation and noise structure.
- The translation from `lift` to `alpha_max` is heuristic. 
  presumably has a principled mapping in their proprietary code.
- Only Meta is calibrated. Real production has multiple calibration
  signals across channels.
