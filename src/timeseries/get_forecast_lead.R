library(ranger)

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

  pred_test_vec <- as.numeric(pred_test_all)
  list(
    rf = mean(pred_test_vec),
    hrf = as.numeric(pred_test_vec %*% w)
  )
}
