# Purpose: Referee 2, Major Comment 3 -- a data-generating process with a
# known ground truth, used to decompose HRF's gain over RF into a
# "structural advantage" (oracle Sigma, mu) and an "estimation effect"
# (feasible Sigma-hat, mu-hat vs. oracle). Uses the classic Friedman (1991)
# nonlinear regression function: x1-x5 are relevant (with two interaction /
# nonlinear terms), x6, ..., xd are pure noise features irrelevant to the
# response, so the design also supports a higher-dimensional / sparse-signal
# regime (Referee 2, Minor Comment 4) by increasing d.
simulate_friedman1 <- function(n, d = 10, noise_sd = 1) {
  stopifnot(d >= 5)
  X <- matrix(runif(n * d), nrow = n, ncol = d)
  f <- 10 * sin(pi * X[, 1] * X[, 2]) +
    20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] +
    5 * X[, 5]
  y <- f + rnorm(n, sd = noise_sd)
  data <- as.data.frame(X)
  colnames(data) <- paste0("x", seq_len(d))
  data$target <- y
  data
}
