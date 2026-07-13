// Bayesian Marketing Mix Model with geometric adstock and Hill saturation.
// Five paid channels + newsletter + trend, seasonality, competitor, events.

data {
  int<lower=1> N;                       // weeks
  int<lower=1> C;                       // paid channels
  matrix<lower=0>[N, C] X_paid;         // weekly spend per paid channel
  vector<lower=0>[N] X_nl;              // newsletter activity
  vector[N] competitor;                 // competitor sales
  array[N] int<lower=0, upper=1> events; // holiday flag
  vector[N] t_idx;                      // 0, 1, ..., N-1
  vector[N] s_cos;                      // cos(2 pi t / 52)
  vector[N] s_sin;                      // sin(2 pi t / 52)
  vector[N] revenue;                    // target
}

parameters {
  // Intercept and controls
  real beta0;
  real beta_trend;
  real beta_cos;
  real beta_sin;
  real beta_comp;
  real beta_ev;

  // Paid channel MMM parameters
  vector<lower=0, upper=1>[C] lambda;     // geometric adstock decay
  vector<lower=0>[C] K;                   // Hill half-saturation point
  vector<lower=0>[C] alpha_max;           // channel max revenue contribution

  // Newsletter (owned channel)
  real<lower=0> alpha_nl;
  real<lower=0, upper=1> lambda_nl;
  real<lower=0> K_nl;

  // Observation noise
  real<lower=0> sigma;
}

transformed parameters {
  matrix[N, C] adstock;
  matrix[N, C] sat;
  vector[N] nl_ad;
  vector[N] nl_sat;
  vector[N] mu;

  // Geometric adstock per paid channel
  for (c in 1:C) {
    adstock[1, c] = X_paid[1, c];
    for (n in 2:N) {
      adstock[n, c] = X_paid[n, c] + lambda[c] * adstock[n - 1, c];
    }
    for (n in 1:N) {
      sat[n, c] = adstock[n, c] / (K[c] + adstock[n, c]);
    }
  }

  // Newsletter adstock + saturation
  nl_ad[1] = X_nl[1];
  for (n in 2:N) {
    nl_ad[n] = X_nl[n] + lambda_nl * nl_ad[n - 1];
  }
  for (n in 1:N) {
    nl_sat[n] = nl_ad[n] / (K_nl + nl_ad[n]);
  }

  // Mean revenue
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
  // Priors
  beta0      ~ normal(250000, 80000);
  beta_trend ~ normal(0, 500);
  beta_cos   ~ normal(0, 50000);
  beta_sin   ~ normal(0, 50000);
  beta_comp  ~ normal(0, 1);
  beta_ev    ~ normal(0, 50000);

  lambda    ~ beta(2, 4);                // adstock decay, mild prior favoring < 0.5
  K         ~ normal(50000, 60000);      // half-normal via <lower=0> constraint
  alpha_max ~ normal(120000, 120000);    // half-normal, wide to allow ROAS > 1 solutions

  alpha_nl  ~ normal(6000, 5000);
  lambda_nl ~ beta(2, 4);
  K_nl      ~ normal(5000, 5000);

  sigma     ~ normal(0, 30000);          // half-normal via <lower=0> constraint

  // Likelihood
  revenue ~ normal(mu, sigma);
}

generated quantities {
  // Per-channel contribution per week (for decomposition plot and ROAS)
  matrix[N, C] contribution;
  for (c in 1:C) {
    for (n in 1:N) {
      contribution[n, c] = alpha_max[c] * sat[n, c];
    }
  }
  vector[N] revenue_pred;
  for (n in 1:N) {
    revenue_pred[n] = normal_rng(mu[n], sigma);
  }
}
