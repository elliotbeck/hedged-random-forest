source("src/utils/get_weights.R")
source("src/cov_estimators/get_cov_ewma_shrink.R")
source("src/timeseries/get_forecast_lead.R")

set.seed(7)
n <- 100
d <- 15
X <- as.data.frame(matrix(rnorm(n * d), n, d))
colnames(X) <- paste0("f", 1:d)
y <- X$f1 * 0.5 + rnorm(n, sd = 0.1)
X_test <- X[n, , drop = FALSE]

result <- get_forecast_lead(X[1:(n - 1), , drop = FALSE], y[1:(n - 1)], X_test,
  num_trees = 50, kappa = 2, ewma_lambda = 0.15, ewma_H = 6
)

stopifnot(is.list(result))
stopifnot(all(c("rf", "hrf") %in% names(result)))
stopifnot(is.numeric(result$rf) && length(result$rf) == 1)
stopifnot(is.numeric(result$hrf) && length(result$hrf) == 1)
stopifnot(is.finite(result$rf) && is.finite(result$hrf))

## both forecasts should be in a plausible range given y's scale (not exploding)
stopifnot(abs(result$rf) < 5)
stopifnot(abs(result$hrf) < 5)

cat("ALL PASS\n")
