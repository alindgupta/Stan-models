/**
 * Mixture of Gaussians model
 */

data {
  int<lower=0> N;  // number of data points
  int<lower=0> K;  // number of Gaussians
  int<lower=0> J;  // number of predictors
  vector[N] y;  // response
  vector[K] htheta;  // hyperparameter vector for Dirichlet
  real<lower=0> hvar;  // hyperparameter for prior variances
}

parameters {
  ordered[K] mu;  // locations of Gaussians
  vector<lower=0>[K] sigma;  // scales of Gaussians
  simplex[K] theta;  // assignment probabilities
}

model {
  real ps[K];

  mu[1] ~ normal(0.3, hvar);
  mu[2] ~ normal(0.8, hvar);
  mu[3] ~ normal(1.4, hvar);

  sigma ~ normal(0, 0.1);
  theta ~ dirichlet(htheta);

  for (n in 1:N) {
    for (k in 1:K) {
      ps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}

generated quantities {
    vector[K] mu_pred;
    for (k in 1:K) {
        mu_pred[k] = normal_rng(mu[k], sigma[k]);
    }
}
