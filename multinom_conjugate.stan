// multinomial likelihood
// logistic normal prior
// uniform priors on mu, half-cauchy on cholesky 

data {
    int<lower=0> r;
    int<lower=0> c;
    int y[r,c];
    vector[c] alpha;
}
parameters {
    simplex[c] theta[r];
}
model {
    for (i in 1:r) {
        target += dirichlet_lpdf(theta[i] | alpha);
        target += multinomial_lpmf(y[i,] | theta[i]);
    } 
}
