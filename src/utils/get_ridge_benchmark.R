source("src/utils/get_mse.R")

ridge <- function(rf_predictions_train_all,
                  train_data,
                  rf_predictions_test_all,
                  test_data,
                  norm_param) {
  X <- rf_predictions_train_all
  y <- train_data$target
  n <- nrow(X)

  # SVD for efficient AIC computation across lambda grid
  svd_X <- svd(X)
  U <- svd_X$u
  d <- svd_X$d
  V <- svd_X$v
  Uty <- drop(t(U) %*% y)

  lambdas <- exp(seq(log(1e-4), log(1e4), length.out = 100))

  aic_values <- sapply(lambdas, function(lambda) {
    scale <- d^2 / (d^2 + lambda)
    fitted <- U %*% (scale * Uty)
    rss <- sum((y - fitted)^2)
    df  <- sum(scale)
    n * log(rss / n) + 2 * df
  })

  lambda_opt <- lambdas[which.min(aic_values)]

  beta_hat <- V %*% ((d / (d^2 + lambda_opt)) * Uty)
  preds <- rf_predictions_test_all %*% beta_hat
  preds <- (preds * norm_param$sd) + norm_param$mean
  mse(preds, test_data$target)
}
