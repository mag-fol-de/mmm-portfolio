# Time-Varying Coefficient MMM

Third project in the MMM portfolio. Focus: time-varying channel effectiveness.
can model channel effectiveness as drifting over time, not fixed.

## Goal

Fit a Bayesian MMM where one channel's max effect (`alpha_max_search`) is
allowed to evolve as a random walk over the 156-week period. The other four
paid channels stay static. Recover the true ramp embedded in the synthetic
data.

## Motivation

Static MMMs assume channel effectiveness is constant. In reality:

- iOS 14 tracking changes shifted attribution between channels.
- TikTok went from niche to dominant in 2-3 years.
- Search demand grows as a brand matures.
- Macroeconomic shocks change consumer responsiveness.

A static MMM averages over all these regimes. A time-varying-coefficient
(TVC) MMM lets the data tell you when and how each channel's effectiveness
shifts.

the reference does not publicly market TVC modelling. This is the part of the
portfolio that signals "I can extend your stack with something modern".

## Model

For static channels (meta, tiktok, youtube, display):
```
contribution[t, c] = alpha_max_c * sat(adstock(spend[t, c]; lambda_c); K_c)
```

For search, the same form but `alpha_max` drifts:
```
alpha_search[t] = alpha_search[t-1] + tau * eta[t]
eta[t] ~ Normal(0, 1)            // non-centred parameterisation
```

`tau` controls how fast `alpha_search` is allowed to move. A tight `tau`
prior pulls toward a static model; a loose `tau` lets the data drive the
drift.

## Random walk prior in plain words

We are saying: "between any two consecutive weeks, the alpha_max change is
roughly Normal(0, tau)". Smooth drift but no large jumps. The data tells us
how big `tau` actually is.

## Data

`data/marketing_data.csv`: 156 weeks. Search's true `alpha_max` ramps from
18,000 (week 0) to 38,000 (week 155). Other channels are static.

Ground truth in `data/ground_truth_static.csv` (fixed channels) and
`data/ground_truth_search_alpha_t.csv` (search ramp).

## How to run

```bash
uv run python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
uv run python generate_data.py
uv run python fit_tvc.py
```

## Outputs

- `figures/01_alpha_search_tvc.png`: posterior mean and 90% CI of
  `alpha_search(t)`, with the true ramp overlaid.
- `figures/02_tvc_vs_static.png`: TVC posterior next to what a static
  model would imply (a flat line around the period mean).
- `samples/posterior_summary.csv`: parameter table.

## Concepts to defend

- **Why a random walk?** Smoothest possible prior on a drifting process
  with no specific shape assumed. Robust to many forms of drift.
- **Why non-centred parameterisation?** Funnel-shaped posteriors make
  HMC-NUTS struggle. Sampling on standardised `eta` and rescaling by `tau`
  decouples the two.
- **How does `tau` get identified?** Through the data. Tighter prior on
  `tau` enforces more smoothness; loose prior lets the model wiggle.
  Posterior on `tau` tells you how much real drift exists.
- **Trade-off with static MMM:** more parameters means more risk of
  overfitting. TVC pays off when the drift is real and large; for a stable
  channel, it just adds noise.
- **Extension:** make all five channels time-varying, or add structural
  break points (regime-switching) for known events like iOS 14.

## Limitations

- Only one channel is time-varying. A full TVC MMM would let all channels
  drift, with potentially correlated innovations.
- The Hill saturation (`K_search`) is held static. Could be made time-varying
  too if the data supports it.
- Random walk is one of many options. Gaussian Process priors, splines,
  and ARMA-driven priors are more flexible but harder to identify.
- Synthetic ramp is monotone and smooth. Real drift can be jumpy.

## Why this lands in a the reference interview

Stan, HMC-NUTS, non-centred parameterisation, prior trade-offs, posterior
calibration against ground truth. You demonstrate the modelling fluency
they screen for, while extending the static MMM they have publicly
described.
