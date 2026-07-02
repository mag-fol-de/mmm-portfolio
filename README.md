# MMM Portfolio

A collection of small projects on Bayesian methods for marketing
measurement, built end-to-end on synthetic data. Each project stands
alone but they share a common target: the Marketing Mix Modelling (MMM)
and geo-lift workflow used by modern SaaS attribution platforms.

## Projects

**[01-bayesian-mmm-stan](./01-bayesian-mmm-stan)**
Bayesian Marketing Mix Model in Stan. Recovers known channel parameters
(adstock, saturation, ROAS) from synthetic weekly retail data.

**[02-geo-lift-calibration](./02-geo-lift-calibration)**
Bayesian geo-lift estimation whose posterior on the incremental effect
feeds back as an informative prior into the MMM. Closes the loop
between experiments and modelling.

**[03-time-varying-mmm](./03-time-varying-mmm)**
Time-varying coefficient MMM. Lets channel effectiveness drift over
time rather than assuming a static ROAS.

**[04-bayesian-sc-sdid](./04-bayesian-sc-sdid)**
Bayesian Synthetic Control vs Bayesian Synthetic Difference-in-Differences
for geo-lift measurement. Comparison of two related causal-inference
methods on the same simulated panel.

## Stack

- Python for pipelines and post-processing
- Stan via cmdstanpy for Bayesian inference
- HMC-NUTS for sampling
- Matplotlib for diagnostics
- uv for dependency management

## Running a project

Each subfolder has its own README with a run recipe. Typical flow:

```bash
cd 04-bayesian-sc-sdid
python generate_data.py
python fit_compare.py
```

## Data note

All datasets are synthetic and generated with fixed seeds for
reproducibility. There is no real customer or campaign data in any
project.
