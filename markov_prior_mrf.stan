data {
  int<lower=0> J;  // number of rows (pixels)
  int<lower=0> K;  // number of columns
  int<lower=0> L;
  real<lower=0> x[J,K];  // training data
  real<lower=0> pvar;  // prior's variance
  real<lower=0> y[J,L];  // testing data
  real<lower=0> z[J,K];  // resampled training data to normalize loglik
  real init;
}

parameters {
  real<lower=0> mu[J];
  real<lower=0> sigma[J];
}

model {
  // priors
  for (j in 1:J) {
    target += normal_lpdf(sigma[j] | 0, 0.1);
  }
  
  target += normal_lpdf(mu[1] | init, pvar);
  for (j in 2:J) {
    target += normal_lpdf(mu[j] | mu[j-1], pvar);
  }
  
  // likelihood
  for (k in 1:K) {
    for (j in 1:J) {
      target += normal_lpdf(x[j,k] | mu[j], sigma[j]);
    }
  }
}

generated quantities {
  real log_lik_test[J,L];
  real log_lik_train[J,K];

  for (k in 1:L) {
    for (j in 1:J) {
      log_lik_test[j,k] = normal_lpdf(y[j,k] | mu[j], sigma[j]);
    }
  }
  
  for (k in 1:K) {
    for (j in 1:J) {
      log_lik_train[j,k] = normal_lpdf(z[j,k] | mu[j], sigma[j]);
    }
  }
}

