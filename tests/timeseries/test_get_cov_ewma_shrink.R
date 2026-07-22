source("src/cov_estimators/get_cov_ewma_shrink.R")

set.seed(42)
Tn <- 60
p <- 5
R <- matrix(rnorm(Tn * p, mean = 0, sd = 1), Tn, p)

result <- get_cov_ewma_shrink(R, lambda = 0.15, H = 6)

stopifnot(is.numeric(result$mu))
stopifnot(length(result$mu) == p)
stopifnot(is.matrix(result$sigma))
stopifnot(all(dim(result$sigma) == c(p, p)))

## symmetry
stopifnot(isTRUE(all.equal(result$sigma, t(result$sigma))))

## diagonal must be strictly positive (variances)
stopifnot(all(diag(result$sigma) > 0))

## PSD: all eigenvalues >= -1e-8 (allow tiny numerical slack)
eig <- eigen(result$sigma, symmetric = TRUE, only.values = TRUE)$values
stopifnot(all(eig > -1e-8))

## shrinkage intensity must strictly shrink toward the constant-variance-covariance
## target: since R is i.i.d. noise here, sigma should differ from the raw (unshrunk)
## weighted sample covariance -- i.e. shrinkage must have actually moved the estimate.
w <- lambda_weights <- 0.15 * (1 - 0.15)^((Tn - 1):0)
xbar <- colMeans(R)
Yc <- sweep(R, 2, xbar, "-")
sigma_hat_unshrunk <- matrix(0, p, p)
for (t in 1:Tn) sigma_hat_unshrunk <- sigma_hat_unshrunk + w[t] * outer(Yc[t, ], Yc[t, ])
stopifnot(!isTRUE(all.equal(result$sigma, sigma_hat_unshrunk)))

cat("ALL PASS\n")
