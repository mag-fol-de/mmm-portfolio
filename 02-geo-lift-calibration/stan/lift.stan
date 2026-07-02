// Step 1: Bayesian geo-lift estimation.
// Difference-in-differences style. Estimate the incremental ROAS on the
// treated channel's extra spend during the experiment.
//
// revenue[i] = region_mean + slope * baseline_meta_spend + lift * extra_meta_spend + eps

data {
  int<lower=1> N;                                    // total rows (regions x weeks)
  int<lower=1, upper=2> R;                           // number of regions
  array[N] int<lower=1, upper=R> region;             // region index per row
  vector<lower=0>[N] baseline_meta;                  // always-on Meta spend
  vector<lower=0>[N] extra_meta;                     // experimental boost (0 in pre-period and for control)
  vector[N] revenue;
}

parameters {
  vector[R] mu_region;                               // region intercepts
  real<lower=0> slope_baseline;                      // baseline Meta-on-revenue slope
  real<lower=0> lift;                                // incremental ROAS on extra Meta spend
  real<lower=0> sigma;
}

model {
  // Priors
  mu_region        ~ normal(200000, 100000);
  slope_baseline   ~ normal(0.3, 0.3);
  lift             ~ normal(0.5, 0.5);   // weakly informative, positive
  sigma            ~ normal(0, 20000);

  // Likelihood
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = mu_region[region[i]]
          + slope_baseline * baseline_meta[i]
          + lift * extra_meta[i];
  }
  revenue ~ normal(mu, sigma);
}
