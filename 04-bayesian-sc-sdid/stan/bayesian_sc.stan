// Bayesian Synthetic Control.
//
// Model the treated unit's pre-period outcome as a weighted average of
// control units' outcomes. Dirichlet prior places the weights on the simplex.
//
//   Y_treated[t] | omega, sigma ~ Normal(Y_controls[t] . omega, sigma)  for t = 1..T0
//
// Post-period counterfactual is the same weighted average extended forward.
// ATT[t] = Y_treated_observed[t] - counterfactual[t].

data {
  int<lower=1> T;                                    // total weeks
  int<lower=1> T0;                                   // last pre-treatment week
  int<lower=1> N;                                    // number of control units
  vector[T] Y_treated;                               // treated unit outcomes
  matrix[T, N] Y_control;                            // control unit outcomes
}

transformed data {
  real sigma_emp = sd(Y_treated[1:T0]);
}

parameters {
  simplex[N] omega;                                  // unit weights, Dirichlet prior
  real<lower=0> sigma;                               // observation noise
}

model {
  omega ~ dirichlet(rep_vector(1.0, N));             // uniform on simplex
  sigma ~ normal(0, sigma_emp);                      // half-normal via lower=0

  // Likelihood, pre-period only
  Y_treated[1:T0] ~ normal(Y_control[1:T0, ] * omega, sigma);
}

generated quantities {
  vector[T] counterfactual;
  vector[T - T0] att_t;
  real att_mean;

  // Pre-period counterfactual: deterministic dot-product (used only for
  // diagnostic plots, not for uncertainty of ATT).
  for (t in 1:T0) {
    counterfactual[t] = dot_product(Y_control[t, ], omega);
  }

  // Post-period counterfactual: posterior predictive draw so ATT
  // uncertainty reflects both weight and observation noise. Using the
  // deterministic dot-product would under-cover the true ATT.
  for (t in (T0 + 1):T) {
    counterfactual[t] = normal_rng(dot_product(Y_control[t, ], omega), sigma);
  }

  // Treatment effect per post-period week
  for (k in 1:(T - T0)) {
    att_t[k] = Y_treated[T0 + k] - counterfactual[T0 + k];
  }
  att_mean = mean(att_t);
}
