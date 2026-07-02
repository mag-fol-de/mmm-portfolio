// Step 2: Bayesian MMM with informative prior on Meta's alpha_max derived
// from the geo-lift experiment posterior. Same structure as project 1's
// MMM, but Meta's alpha_max prior is centred on the lift estimate.

data {
  int<lower=1> N;
  int<lower=1> C;
  matrix<lower=0>[N, C] X_paid;
  vector<lower=0>[N] X_nl;
  vector[N] competitor;
  array[N] int<lower=0, upper=1> events;
  vector[N] t_idx;
  vector[N] s_cos;
  vector[N] s_sin;
  vector[N] revenue;

  // Calibration prior on Meta (channel index meta_idx)
  int<lower=1, upper=10> meta_idx;
  real<lower=0> prior_alpha_max_meta_mean;
  real<lower=0> prior_alpha_max_meta_sd;
}

parameters {
  real beta0;
  real beta_trend;
  real beta_cos;
  real beta_sin;
  real beta_comp;
  real beta_ev;

  vector<lower=0, upper=1>[C] lambda;
  vector<lower=0>[C] K;
  vector<lower=0>[C] alpha_max;

  real<lower=0> alpha_nl;
  real<lower=0, upper=1> lambda_nl;
  real<lower=0> K_nl;

  real<lower=0> sigma;
}

transformed parameters {
  matrix[N, C] adstock;
  matrix[N, C] sat;
  vector[N] nl_ad;
  vector[N] nl_sat;
  vector[N] mu;

  for (c in 1:C) {
    adstock[1, c] = X_paid[1, c];
    for (n in 2:N) {
      adstock[n, c] = X_paid[n, c] + lambda[c] * adstock[n - 1, c];
    }
    for (n in 1:N) {
      sat[n, c] = adstock[n, c] / (K[c] + adstock[n, c]);
    }
  }

  nl_ad[1] = X_nl[1];
  for (n in 2:N) {
    nl_ad[n] = X_nl[n] + lambda_nl * nl_ad[n - 1];
  }
  for (n in 1:N) {
    nl_sat[n] = nl_ad[n] / (K_nl + nl_ad[n]);
  }

  for (n in 1:N) {
    mu[n] = beta0
          + beta_trend * t_idx[n]
          + beta_cos * s_cos[n]
          + beta_sin * s_sin[n]
          + beta_comp * competitor[n]
          + beta_ev * events[n]
          + alpha_nl * nl_sat[n];
    for (c in 1:C) {
      mu[n] += alpha_max[c] * sat[n, c];
    }
  }
}

model {
  beta0      ~ normal(250000, 80000);
  beta_trend ~ normal(0, 500);
  beta_cos   ~ normal(0, 50000);
  beta_sin   ~ normal(0, 50000);
  beta_comp  ~ normal(0, 1);
  beta_ev    ~ normal(0, 50000);

  lambda ~ beta(2, 4);
  K      ~ normal(40000, 40000);

  // alpha_max priors. Meta gets the calibration prior, the rest stay weakly
  // informative.
  for (c in 1:C) {
    if (c == meta_idx) {
      alpha_max[c] ~ normal(prior_alpha_max_meta_mean, prior_alpha_max_meta_sd);
    } else {
      alpha_max[c] ~ normal(30000, 30000);
    }
  }

  alpha_nl  ~ normal(6000, 5000);
  lambda_nl ~ beta(2, 4);
  K_nl      ~ normal(5000, 5000);

  sigma     ~ normal(0, 30000);

  revenue ~ normal(mu, sigma);
}
