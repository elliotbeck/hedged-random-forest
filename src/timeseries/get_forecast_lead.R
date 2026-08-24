library(ranger)
source("src/utils/get_minvar_benchmark.R")

get_forecast_lead <- function(features_train, label_train, features_test,
                               num_trees = 500, kappa = 2,
                               ewma_lambda = 0.15, ewma_H = 6) {
  mtry <- max(1, round(ncol(features_train) / 3))

  rf <- ranger(x = features_train, y = label_train, num.trees = num_trees, mtry = mtry, keep.inbag = TRUE)

  pred_train_all <- predict(rf, features_train, predict.all = TRUE)$predictions
  pred_test_all <- predict(rf, features_test, predict.all = TRUE)$predictions

  R <- pred_train_all - label_train # in-sample tree residual matrix, chronological order

  mu_sigma <- get_cov_ewma_shrink(R, lambda = ewma_lambda, H = ewma_H)
  w <- get_weights(mu_sigma$mu, mu_sigma$sigma, kappa)

  # HRF-MinVar (Referee 1, Comments 3-4, applied to the time-series setting):
  # drop the bias term (w'mu)^2 from the optimization, minimizing variance
  # alone, then add back the systematic bias w'mu post-hoc, exactly as in the
  # i.i.d. benchmark of src/utils/get_minvar_benchmark.R.
  w_minvar <- get_minvar_weights(mu_sigma$sigma, kappa)
  bias_correction <- as.numeric(t(w_minvar) %*% mu_sigma$mu)

  pred_test_vec <- as.numeric(pred_test_all)
  list(
    rf = mean(pred_test_vec),
    hrf = as.numeric(pred_test_vec %*% w),
    hrf_minvar = as.numeric(pred_test_vec %*% w_minvar) + bias_correction
  )
}
