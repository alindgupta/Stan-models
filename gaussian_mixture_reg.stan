/* Mixture of Gaussians model
 * 
 * Note: for inference of posteriors in a mixture model,
 * the priors need to be non-exchangeable
 *
 */


data {
  int<lower=0> N;  // number of data points
  int<lower=0> K;  // number of Gaussians
  int<lower=0> J;  // number of predictors
  vector[N] y;  // response
  vector[K] htheta;  // hyperparameter vector for Dirichlet
  real<lower=0> hvar;  // hyperparameter for prior variances
  matrix[N,J] X;  // model matrix 
}

/*

For each of the K Gaussians:
    Yi = bi0 + biX
 */

parameters {
  vector[K] bs;  // regression weights for `SigD`
  vector[K] bm;  // regression weights for `Mature`
  vector[K] b0;  // regression biases
  ordered[K] mu;  // locations of Gaussians
  vector<lower=0>[K] sigma;  // scales of Gaussians
  simplex[K] theta;  // assignment probabilities
}

transformed parameters {
  vector[N] y_reg;
  for (n in 1:N) {
    y_reg[n] = b0 + bs * X[n, 1] + bm * X[n, 2];
  }
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
    target += log_sum_exp(ps);  // smooth approximation of max()
  }
}

generated quantities {
    vector[K] mu_pred;
    for (k in 1:K) {
        mu_pred[k] = normal_rng(mu[k], sigma[k]);
    }
}

/*
generated quantities {
  vector[K] mu_pred;
  int w_pred[K];
  real y_pred;
  vector[N] log_lik;

  for (k in 1:K) {
    mu_pred[k] = normal_rng(mu[k], sigma[k]);
  }
  w_pred = multinomial_rng(theta, 3);
  
  y_pred = mu_pred[1] * w_pred[1] + mu_pred[2] * w_pred[2] + mu_pred[3] * w_pred[3];

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | y_pred, hvar);
  }
}
*/
