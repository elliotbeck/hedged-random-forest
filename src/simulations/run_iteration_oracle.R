# Purpose: Referee 2, Major Comment 3 -- one repetition of the
# oracle-vs-feasible decomposition. A single, realistically-sized (n_obs)
# forest is grown; its HRF weights are estimated three ways, all as
# residuals of these SAME p trees:
#   - in-sample, plain sample covariance (in-sample residuals)
#   - in-sample, NLS/QIS shrinkage covariance (in-sample residuals) -- the
#     paper's current main specification
#   - oracle: residuals of these SAME p trees on a large, separate,
#     out-of-sample set (e.g. 50k rows), standing in for the true (mu,
#     Sigma) for these specific trees
# Every resulting weight vector, plus the equal-weighted RF benchmark, is
# scored the same way: form forecasts from this SAME forest on a further,
# independent test sample and compare them to the true outcomes there.
library(ranger)
source("src/utils/get_mse.R")
source("src/utils/get_weights.R")
source("src/cov_estimators/get_cov_qis.R")
source("src/simulations/simulate_friedman1.R")

run_iteration_oracle <- function(n_obs,
                                  num_trees = 500,
                                  kappa = 2,
                                  d = 10,
                                  noise_sd = 1,
                                  n_oracle = 50000,
                                  n_test = 20000,
                                  min_node_size = 5) {
  # Single forest, held fixed for the rest of this repetition.
  train_data <- simulate_friedman1(n_obs, d = d, noise_sd = noise_sd)
  rf_model <- ranger::ranger(
    target ~ .,
    data = train_data,
    num.trees = num_trees,
    mtry = floor((ncol(train_data) - 1) / 3),
    replace = TRUE,
    min.node.size = min_node_size
  )

  # In-sample residuals (the paper's current construction).
  preds_train_all <- predict(rf_model, train_data, predict.all = TRUE)$predictions
  resid_train <- train_data$target - preds_train_all
  mu_hat <- colMeans(resid_train)
  Sigma_hat_nls <- get_cov_qis(resid_train)

  w_eq <- rep(1 / num_trees, num_trees)
  w_hat_nls <- get_weights(mu_hat, Sigma_hat_nls, kappa)

  # Plain sample covariance can be singular/ill-conditioned (e.g. n_obs
  # close to or below num_trees); guard accordingly, as elsewhere in the
  # codebase (run_iteration.R).
  w_hat_sample <- tryCatch({
    Sigma_hat_sample <- cov(resid_train)
    get_weights(mu_hat, Sigma_hat_sample, kappa)
  }, error = function(e) rep(NA_real_, num_trees))

  # Oracle (mu, Sigma) for these SAME p trees, from a large, separate
  # out-of-sample set -- essentially exact given the sample size.
  oracle_data <- simulate_friedman1(n_oracle, d = d, noise_sd = noise_sd)
  preds_oracle_all <- predict(rf_model, oracle_data, predict.all = TRUE)$predictions
  resid_oracle <- oracle_data$target - preds_oracle_all
  mu_star <- colMeans(resid_oracle)
  Sigma_star <- cov(resid_oracle)
  w_star <- get_weights(mu_star, Sigma_star, kappa)

  # Fresh, independent test sample: literal forecasts vs. true outcomes,
  # using this SAME forest throughout.
  test_data <- simulate_friedman1(n_test, d = d, noise_sd = noise_sd)
  preds_test_all <- predict(rf_model, test_data, predict.all = TRUE)$predictions

  forecast_mse <- function(w) mse(preds_test_all %*% w, test_data$target)

  rf <- forecast_mse(w_eq)
  hrf_oracle <- forecast_mse(w_star)
  hrf_nls <- forecast_mse(w_hat_nls)
  hrf_sample <- if (anyNA(w_hat_sample)) NA_real_ else forecast_mse(w_hat_sample)

  # The referee's decomposition, computed explicitly:
  #   HRF(Sigma-hat) - RF = [HRF(Sigma) - RF] + [HRF(Sigma-hat) - HRF(Sigma)]
  #                          structural advantage    estimation effect
  structural_advantage <- hrf_oracle - rf
  estimation_effect_sample <- hrf_sample - hrf_oracle
  estimation_effect_nls <- hrf_nls - hrf_oracle

  c(
    n_obs = n_obs,
    rf = rf,
    hrf_oracle = hrf_oracle,
    hrf_sample = hrf_sample,
    hrf_nls = hrf_nls,
    structural_advantage = structural_advantage,
    estimation_effect_sample = estimation_effect_sample,
    estimation_effect_nls = estimation_effect_nls,
    total_effect_sample = hrf_sample - rf,
    total_effect_nls = hrf_nls - rf
  )
}
