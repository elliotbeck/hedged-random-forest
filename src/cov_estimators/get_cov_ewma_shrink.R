get_cov_ewma_shrink <- function(R, lambda = 0.15, H = 6) {
  Tn <- nrow(R)
  p <- ncol(R)
  w <- lambda * (1 - lambda)^((Tn - 1):0)

  ## --- EWMA mean (D.1) ---
  mu_hat <- as.numeric(colSums(R * w))

  ## --- EWMA covariance (D.2), centered on the SIMPLE mean, per the paper ---
  xbar <- colMeans(R)
  Y <- sweep(R, 2, xbar, "-")
  Sigma_hat <- matrix(0, p, p)
  for (t in seq_len(Tn)) Sigma_hat <- Sigma_hat + w[t] * outer(Y[t, ], Y[t, ])

  ## --- shrinkage target for Sigma: constant-variance-covariance matrix F (D.4) ---
  f1 <- mean(diag(Sigma_hat))
  f2 <- mean(Sigma_hat[upper.tri(Sigma_hat)])
  F_target <- matrix(f2, p, p)
  diag(F_target) <- f1

  ## --- data-dependent shrinkage intensity for Sigma (D.6-D.8), vectorized ---
  D_list <- vector("list", Tn)
  for (t in seq_len(Tn)) D_list[[t]] <- outer(Y[t, ], Y[t, ]) - Sigma_hat

  psi_sum <- function(h) {
    if (h == 0) {
      s <- 0
      for (t in seq_len(Tn)) s <- s + sum(D_list[[t]] * D_list[[t]])
    } else {
      s <- 0
      for (t in (h + 1):Tn) s <- s + sum(D_list[[t]] * D_list[[t - h]])
    }
    s / Tn
  }

  nu_sigma <- (lambda^2 / (1 - (1 - lambda)^2)) *
    (psi_sum(0) + 2 * sum(sapply(seq_len(H), function(h) (1 - lambda)^h * psi_sum(h))))
  gamma_sigma <- sum((F_target - Sigma_hat)^2)
  alpha_sigma <- nu_sigma / (nu_sigma + gamma_sigma)
  Sigma_shrunk <- alpha_sigma * F_target + (1 - alpha_sigma) * Sigma_hat

  ## --- shrinkage target for mu: constant-mean vector (D.5) ---
  mu_star <- mean(mu_hat)

  ## --- data-dependent shrinkage intensity for mu (D.9), cheap per-column acf ---
  nu_i <- sapply(seq_len(p), function(i) {
    psi_i <- as.numeric(acf(R[, i], lag.max = H, type = "covariance", demean = TRUE, plot = FALSE)$acf)
    (lambda^2 / (1 - (1 - lambda)^2)) * (psi_i[1] + 2 * sum((1 - lambda)^(1:H) * psi_i[2:(H + 1)]))
  })
  nu_mu <- sum(nu_i)
  gamma_mu <- sum((mu_star - mu_hat)^2)
  alpha_mu <- nu_mu / (nu_mu + gamma_mu)
  mu_shrunk <- alpha_mu * mu_star + (1 - alpha_mu) * mu_hat

  list(mu = mu_shrunk, sigma = Sigma_shrunk)
}
