# Bayesian Synthetic Control vs Bayesian Synthetic Diff-in-Diff

Fourth project in the MMM portfolio. Focus: geo-lift estimation via
Bayesian causal inference, comparing two related methods on the same
simulated panel.

## Goal

Implement BSC and BSDID in Stan, fit both to the same simulated geo-panel
with a known true ATT, and compare ATT posterior bias, CI width, and
counterfactual fit.

## Data

Simulated panel: 21 units (1 treated + 20 controls), 40 weeks (30 pre + 10
post). Known per-week treatment effect `tau = 25,000` SEK on the treated
unit during the post-period. Shared trend and annual seasonality, unit
baselines drawn Uniform(150k, 350k), Gaussian noise sigma 8,000.

Ground truth in `data/ground_truth.csv`.

## Methods

### Bayesian Synthetic Control (BSC)

Unit-weighted average of controls. Weights live on the simplex, prior
Dirichlet(1):

```
Y_treated[t] | omega, sigma  ~ Normal(Y_control[t] . omega, sigma)   for t <= T0
omega                        ~ Dirichlet(1, ..., 1)
sigma                        ~ HalfNormal(sd(Y_treated_pre))
```

Posterior on omega drives the counterfactual. ATT[t] = treated[t] - cf[t]
for t > T0. ATT_mean averaged over the post-period.

### Bayesian Synthetic Difference-in-Differences (BSDID)

Adds time weights on the pre-period to net out any persistent gap between
the treated unit and its synthetic counterfactual. Both weight sets are
simplices with Dirichlet(1) priors, and lambda is additionally anchored
to data via a likelihood on the controls' pre-vs-post levels:

```
omega   ~ Dirichlet(1)
lambda  ~ Dirichlet(1)
Y_control_post_mean[i] ~ Normal(sum_t lambda[t] * Y_control[t, i], sigma_lambda)
ATT_t   = (Y_treated[t] - omega . Y_control[t])
        - sum_s lambda[s] * (Y_treated[s] - omega . Y_control[s])      for t > T0, s <= T0
```

The second term is the lambda-weighted pre-period gap. Subtracting it
removes any persistent bias between treated and synthetic control that
would otherwise leak into the ATT estimate.

## How to run

```bash
uv run python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"
uv run python generate_data.py
uv run python fit_compare.py
```

## Outputs

| File | Content |
|---|---|
| `figures/01_att_posteriors.png` | ATT posterior densities, BSC vs BSDID vs truth |
| `figures/02_counterfactual_sc.png` | Treated unit vs BSC counterfactual over time |
| `figures/03_counterfactual_sdid.png` | Treated unit vs BSDID counterfactual over time |
| `figures/04_unit_weights.png` | Posterior mean omega per control unit, BSC vs BSDID |
| `figures/05_time_weights.png` | Posterior mean lambda per pre-period week (BSDID only) |
| `figures/06_att_path.png` | Per-week ATT posterior with 90% CI |
| `samples/att_comparison.csv` | Side-by-side ATT mean, 90% CI, CI width |

## What is not in scope

- **Informative priors**: I use Dirichlet(1) uniform priors. A production
  model would likely use informative priors tied to geographic similarity,
  market size, or category benchmarks.
- **Continuous treatment intensity**: only on/off treatment is modelled.
  Real geo-lift in marketing has varying spend levels per region.
- **Hierarchical pooling across experiments**: a single experiment in
  isolation. A real system pools effects across multiple lift tests.
- **Cross-channel adjustment**: I treat the experiment as one channel in
  isolation. Cross-channel interactions are out of scope.
- **Smoothness regularisation on lambda**: Arkhangelsky et al. (2021)
  propose a specific zeta-regularisation in the frequentist version. The
  Bayesian equivalent would be a hyperprior on lambda concentration, not
  implemented here.

## References

- **Synthetic Difference-in-Differences (the source for SDID).**
  Arkhangelsky, Athey, Hirshberg, Imbens, and Wager (2021).
  *Synthetic Difference-in-Differences*, American Economic Review,
  111(12), 4088–4118.
  Paper: https://www.aeaweb.org/articles?id=10.1257/aer.20190159 ·
  Working paper: https://arxiv.org/abs/1812.09970 ·
  Reference R implementation: https://github.com/synth-inference/synthdid.

- **Synthetic Control (the source for SC).**
  Abadie, Diamond, and Hainmueller (2010).
  *Synthetic Control Methods for Comparative Case Studies*, JASA.
  https://economics.mit.edu/sites/default/files/publications/Synthetic%20Control%20Methods.pdf.

- **Bayesian Causal Impact (closest existing Bayesian SC).**
  Brodersen, Gallusser, Koehler, Remy, and Scott (2015).
  *Inferring Causal Impact Using Bayesian Structural Time-Series Models*,
  Annals of Applied Statistics, 9(1), 247–274.
  https://research.google/pubs/inferring-causal-impact-using-bayesian-structural-time-series-models/.
