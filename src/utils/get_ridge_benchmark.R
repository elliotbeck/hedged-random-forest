source("src/utils/get_mse.R")

# Constrained ridge (sum-to-one) with AIC-based lambda selection.
# Uses the analytical closed-form solution and exact hat-matrix trace
# for the equality-constrained problem, derived via economy SVD.
constrained_ridge <- function(X_train, y_train, X_test) {
  n <- nrow(X_train)
  p <- ncol(X_train)

  sv  <- svd(X_train)
  U   <- sv$u; d <- sv$d; V <- sv$v
  Uty <- drop(t(U) %*% y_train)
  s   <- colSums(V)       # V'*1_p  (k-vector)
  s2  <- sum(s^2)         # ||V'*1_p||^2  <=  p

  lambdas <- exp(seq(log(1e-4), log(1e4), length.out = 100))

  aic_vals <- sapply(lambdas, function(lam) {
    inv    <- 1 / (d^2 + lam)
    sc     <- d^2 * inv                                   # d^2/(d^2+lam)
    num    <- 1 - sum(s * d * inv * Uty)                  # 1 - 1'*A*X'y
    den    <- sum(s^2 * inv) + (p - s2) / lam             # 1'*A*1
    fitted <- U %*% (sc * Uty) + (num / den) * (U %*% (d * s * inv))
    rss    <- sum((y_train - fitted)^2)
    tr_H   <- sum(sc) - sum((d * s * inv)^2) / den        # tr of hat matrix
    n * log(rss / n) + 2 * tr_H
  })

  lam <- lambdas[which.min(aic_vals)]
  inv <- 1 / (d^2 + lam)
  num <- 1 - sum(s * d * inv * Uty)
  den <- sum(s^2 * inv) + (p - s2) / lam

  # Test predictions: X_test * w*(lam), computed without forming w* explicitly
  XV   <- X_test %*% V
  XV %*% (d * inv * Uty) +
    (num / den) * (XV %*% (s * inv) + (rowSums(X_test) - XV %*% s) / lam)
}

ridge <- function(rf_predictions_train_all,
                  train_data,
                  rf_predictions_test_all,
                  test_data,
                  norm_param,
                  return_preds = FALSE) {
  preds <- constrained_ridge(
    X_train = rf_predictions_train_all,
    y_train = train_data$target,
    X_test  = rf_predictions_test_all
  )
  preds <- (preds * norm_param$sd) + norm_param$mean
  if (return_preds) return(drop(preds))
  mse(preds, test_data$target)
}
