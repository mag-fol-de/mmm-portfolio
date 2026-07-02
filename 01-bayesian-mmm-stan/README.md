# Bayesian MMM in Stan

First project in the MMM portfolio.

## Goal

Build a Bayesian Marketing Mix Model in Stan that recovers known channel
parameters from synthetic weekly retail data. Matches the reference core
modelling stack (Stan with HMC-NUTS).

## Approach

```
spend per channel
   |
   v
geometric adstock (lambda)
   |
   v
Hill saturation (K)
   |
   v
linear contribution (alpha_max)
   |
   +-> sum with trend, season, competitor, events, newsletter, intercept
   |
   v
revenue (Normal likelihood, sigma)
```

All parameters get priors. Stan's HMC-NUTS samples the joint posterior over
(lambda_c, K_c, alpha_max_c) per channel plus all controls. Decomposition
and ROAS are computed per posterior draw, giving credible intervals on
every output.

## What the model captures

- **Adstock (lambda)**: how long the effect of one week of spend persists.
- **Saturation (K, alpha_max)**: diminishing returns to spend, with a
  channel ceiling.
- **Controls**: trend, annual seasonality, competitor pressure, holiday
  events, owned-channel newsletter.

## How to run

Install cmdstan once (needs a C++ compiler; install RTools first on
Windows):

```bash
uv run python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
```

Then fit:

```bash
uv run python fit_mmm.py
```

Runtime: a few minutes on a laptop for 4 chains x 1000 sampling iters.

## Outputs

| File | Content |
|---|---|
| `samples/posterior.nc` | Full ArviZ InferenceData |
| `samples/posterior_summary.csv` | Posterior mean, sd, 90% CI, R-hat, ESS |
| `samples/attribution.csv` | Total revenue contribution per driver |
| `samples/roas.csv` | ROAS per channel with credible interval |
| `figures/01_actual_vs_modeled.png` | Fit overlay with 90% CI |
| `figures/02_decomposition.png` | Per-driver contribution over time |
| `figures/03_roas.png` | ROAS per channel with error bars |

## Calibration story

The synthetic data was generated with known `lambda`, `K`, and `alpha_max`
per channel (`data/ground_truth.csv`). After fitting, the script prints a
table comparing the posterior to those true values. A correctly specified
Bayesian MMM should put the true value inside its 90% credible interval for
most parameters.

This is the same validation pattern the reference applies in practice, except
they calibrate against real lift tests rather than synthetic truth (see
project 2).

## Modelling decisions worth defending

- **Stan over PyMC**: matches the reference stack and gives access to ADVI
  and HMC-NUTS without leaving the same model code.
- **Beta(2, 4) prior on lambda**: mild belief that decay sits below 0.5
  for most channels, but data can pull it up.
- **HalfNormal on K and alpha_max**: positive support, weakly informative.
- **Non-centred parameterisation not used here**: the model is small
  enough that the centred form converges. Switch to non-centred if
  divergences appear.

## Limitations

- Synthetic data, not real client data.
- Fixed lambda and K over time (static MMM). Project 3 relaxes this.
- No external calibration via lift tests. Project 2 adds that layer.
- No budget optimisation on top of the fitted model.
