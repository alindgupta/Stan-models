// multinomial likelihood
// logistic normal prior
// uniform priors on mu, half-cauchy on cholesky 
// initial setting for LKJ prior hyperparameter was 10
// and sigma was (1,1,1)

data {
    int<lower=0> r;
    int<lower=0> c;
    int y[r,c];
    vector<lower=0>[c] sigma;
}
parameters {
    vector<lower=0, upper=1>[c] mu;
    vector[c] psi;
    cholesky_factor_corr[c] Lcorr;
}
transformed parameters {
    simplex[c] theta;
    theta = softmax(psi);
}
model {
    for (i in 1:r) {
        target += multinomial_lpmf(y[i,] | theta);
    } 
    target += multi_normal_cholesky_lpdf(psi | mu, diag_pre_multiply(sigma, Lcorr));
    target += lkj_corr_cholesky_lpdf(Lcorr | 2);
    target += normal_lpdf(mu | 0.39, 0.1); // changed mean to average of extreme values, p(outside 0.211,0.577) = 7%
}
generated quantities {
    int y_pred[c];
    y_pred = multinomial_rng(theta, 15);
}

