// Time-varying coefficient MMM.
// One channel (search by convention) has a random-walk prior on its
// alpha_max[t]. The remaining channels stay static. Adstock decay and
// saturation half-points are static for all channels.
//
// Random walk: alpha_search[t] = alpha_search[t-1] + tau * eta[t], eta ~ N(0,1)
// Non-centred parameterisation for sampling efficiency.

data {
  int<lower=1> N;
  int<lower=1> C;                       // number of static channels (excluding the time-varying one)
  matrix<lower=0>[N, C] X_static;       // static channels: meta, tiktok, youtube, display
  vector<lower=0>[N] X_search;          // time-varying channel
  vector<lower=0>[N] X_nl;
  vector[N] competitor;
  array[N] int<lower=0, upper=1> events;
  vector[N] t_idx;
  vector[N] s_cos;
  vector[N] s_sin;
  vector[N] revenue;
}

parameters {
  real beta0;
  real beta_trend;
  real beta_cos;
  real beta_sin;
  real beta_comp;
  real beta_ev;

  // Static channels
  vector<lower=0, upper=1>[C] lambda;
  vector<lower=0>[C] K;
  vector<lower=0>[C] alpha_max_static;

  // Time-varying search
  real<lower=0, upper=1> lambda_search;
  real<lower=0> K_search;
  real<lower=0> alpha_search_0;          // initial level
  real<lower=0> tau;                     // innovation sd
  vector[N - 1] eta;                     // standardised innovations (non-centred)

  // Newsletter
  real<lower=0> alpha_nl;
  real<lower=0, upper=1> lambda_nl;
  real<lower=0> K_nl;

  real<lower=0> sigma;
}

transformed parameters {
  // Build the random-walk path of alpha_max for search
  vector[N] alpha_search;
  alpha_search[1] = alpha_search_0;
  for (n in 2:N) {
    alpha_search[n] = alpha_search[n - 1] + tau * eta[n - 1];
  }

  // Adstock for static channels
  matrix[N, C] adstock_static;
  matrix[N, C] sat_static;
  for (c in 1:C) {
    adstock_static[1, c] = X_static[1, c];
    for (n in 2:N) {
      adstock_static[n, c] = X_static[n, c] + lambda[c] * adstock_static[n - 1, c];
    }
    for (n in 1:N) {
      sat_static[n, c] = adstock_static[n, c] / (K[c] + adstock_static[n, c]);
    }
  }

  // Adstock + saturation for search
  vector[N] ad_search;
  vector[N] sat_search;
  ad_search[1] = X_search[1];
  for (n in 2:N) {
    ad_search[n] = X_search[n] + lambda_search * ad_search[n - 1];
  }
  for (n in 1:N) {
    sat_search[n] = ad_search[n] / (K_search + ad_search[n]);
  }

  // Newsletter
  vector[N] nl_ad;
  vector[N] nl_sat;
  nl_ad[1] = X_nl[1];
  for (n in 2:N) {
    nl_ad[n] = X_nl[n] + lambda_nl * nl_ad[n - 1];
  }
  for (n in 1:N) {
    nl_sat[n] = nl_ad[n] / (K_nl + nl_ad[n]);
  }

  // Mean
  vector[N] mu;
  for (n in 1:N) {
    mu[n] = beta0
          + beta_trend * t_idx[n]
          + beta_cos * s_cos[n]
          + beta_sin * s_sin[n]
          + beta_comp * competitor[n]
          + beta_ev * events[n]
          + alpha_nl * nl_sat[n]
          + alpha_search[n] * sat_search[n];
    for (c in 1:C) {
      mu[n] += alpha_max_static[c] * sat_static[n, c];
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

  lambda            ~ beta(2, 4);
  K                 ~ normal(40000, 40000);
  alpha_max_static  ~ normal(30000, 30000);

  lambda_search   ~ beta(2, 4);
  K_search        ~ normal(20000, 20000);
  alpha_search_0  ~ normal(25000, 15000);
  tau             ~ normal(0, 2000);   // half-normal via lower=0
  eta             ~ std_normal();      // implies alpha_search[t] - alpha_search[t-1] ~ N(0, tau)

  alpha_nl  ~ normal(6000, 5000);
  lambda_nl ~ beta(2, 4);
  K_nl      ~ normal(5000, 5000);

  sigma     ~ normal(0, 30000);

  revenue ~ normal(mu, sigma);
}
