// Bayesian Synthetic Difference-in-Differences.
//
// Two simplex weight vectors: omega over control units, lambda over
// pre-treatment time periods.
//
//   ATT = (Y_treated_post  - omega . Y_control_post)
//       - lambda . (Y_treated_pre  - omega . Y_control_pre)
//
// Two likelihoods. The first ties omega and sigma to the treated unit's
// pre-period fit. The second ties lambda to each control unit's pre-vs-post
// relationship: the lambda-weighted pre-period level of a control should
// predict that control's post-period mean.

data {
  int<lower=1> T;
  int<lower=1> T0;
  int<lower=1> N;
  vector[T] Y_treated;
  matrix[T, N] Y_control;
}

transformed data {
  real sigma_emp = sd(Y_treated[1:T0]);

  // Mean post-period outcome per control unit (used as data for lambda likelihood)
  vector[N] Y_control_post_mean;
  for (i in 1:N) {
    Y_control_post_mean[i] = mean(Y_control[(T0 + 1):T, i]);
  }
  // Empirical scale for the lambda residual (across-unit sd of post means)
  real sigma_lambda_emp = sd(Y_control_post_mean);
}

parameters {
  simplex[N]  omega;
  simplex[T0] lambda;
  real<lower=0> sigma;
  real<lower=0> sigma_lambda;
}

model {
  omega        ~ dirichlet(rep_vector(1.0, N));
  lambda       ~ dirichlet(rep_vector(1.0, T0));
  sigma        ~ normal(0, sigma_emp);
  sigma_lambda ~ normal(0, sigma_lambda_emp);

  // Likelihood 1: treated unit pre-period (ties omega and sigma to data)
  Y_treated[1:T0] ~ normal(Y_control[1:T0, ] * omega, sigma);

  // Likelihood 2: controls' pre-vs-post (ties lambda to data)
  for (i in 1:N) {
    Y_control_post_mean[i] ~ normal(dot_product(lambda, Y_control[1:T0, i]),
                                    sigma_lambda);
  }
}

generated quantities {
  vector[T] counterfactual;
  vector[T - T0] att_t;
  real att_mean;
  real baseline_diff;

  for (t in 1:T) {
    counterfactual[t] = dot_product(Y_control[t, ], omega);
  }

  baseline_diff = 0;
  for (t in 1:T0) {
    baseline_diff += lambda[t] * (Y_treated[t] - counterfactual[t]);
  }

  for (k in 1:(T - T0)) {
    att_t[k] = (Y_treated[T0 + k] - counterfactual[T0 + k]) - baseline_diff;
  }
  att_mean = mean(att_t);
}
