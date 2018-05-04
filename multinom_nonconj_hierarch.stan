data {
  int<lower=0> J; // number of rows
  int<lower=0> K; // number of columns (equals 3)
  int<lower=0> N; // number of experiments
  int group[J];  // groupings, one for each trial
  int y[J,K];  // data matrix
  vector<lower=0>[K] sigma;  // covariance hyperparameter
}

parameters {
  vector<lower=0, upper=1>[K] psi[N];  // experiment-level effect
  vector<lower=0, upper=1>[K] mu[N];  // experiment-level effect
  cholesky_factor_corr[K] Lcorr;  // correlations
  real<lower=0, upper=1> mu_[K];  // group-level effect
  real<lower=0> var_[K]; 
}

transformed parameters {
  simplex[K] theta[N];
  for (n in 1:N) {
    theta[n,] = softmax(psi[n,]); 
  }
}

model {

  for (n in 1:N) {
    for (i in 1:K) {
      target += normal_lpdf(mu_[i] | 0.3, 0.1);
      target += normal_lpdf(var_[i] | 0, 0.1);
      target += normal_lpdf(mu[n,i] | mu_[i], var_[i]);
    }
    target += multi_normal_cholesky_lpdf(psi[n,] | mu[n,], diag_pre_multiply(sigma, Lcorr));
    target += lkj_corr_cholesky_lpdf(Lcorr | 10);
  }

  for (j in 1:J) {
    target += multinomial_lpmf(y[j,] | theta[group[j],]);
  }
} 

