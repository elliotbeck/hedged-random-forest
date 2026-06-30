run_iteration_owrf <- function(data, num_trees = 100, kappas = c(1, 2),
                               include_owrf     = TRUE,
                               include_ridge    = TRUE,
                               include_ridge2nd = TRUE) {
  n       <- nrow(data)
  n_train <- round(n * 0.7)

  idx        <- sample(seq_len(n))
  train_data <- data[idx[seq_len(n_train)], ]
  test_data  <- data[idx[(n_train + 1):n], ]

  norm_param <- list(
    mean = mean(train_data$target),
    sd   = sd(train_data$target)
  )
  if (norm_param$sd == 0) return(NULL)
  train_data$target <- (train_data$target - norm_param$mean) / norm_param$sd
  # Keep original test labels (unstandardised) for error computation
  y_test <- test_data$target

  # RF: Chen et al. settings — 100 trees, min.node.size = floor(sqrt(n_train))
  rf_model <- ranger::ranger(
    target ~ .,
    data          = train_data,
    num.trees     = num_trees,
    mtry          = floor((ncol(train_data) - 1) / 3),
    replace       = TRUE,
    keep.inbag    = TRUE,
    min.node.size = 5
  )

  rf_preds <- (predict(rf_model, test_data)$predictions * norm_param$sd) + norm_param$mean

  train_all <- predict(rf_model, train_data, predict.all = TRUE)$predictions
  test_all  <- predict(rf_model, test_data,  predict.all = TRUE)$predictions

  residuals      <- train_data$target - train_all
  mean_vector    <- colMeans(residuals)
  cov_matrix_nls <- get_cov_qis(residuals)

  # Predictions for each method (denormalised)
  preds <- list()
  preds[["rf"]] <- rf_preds

  for (k in kappas) {
    w <- get_weights(mean_vector, cov_matrix_nls, kappa = k)
    p <- drop(as.matrix(test_all) %*% w)
    preds[[paste0("hrf_nls_", k)]] <- (p * norm_param$sd) + norm_param$mean
  }

  preds[["wrf"]] <- winham(
    train_data               = train_data,
    test_data                = test_data,
    rf_model                 = rf_model,
    rf_predictions_train_all = train_all,
    rf_predictions_test_all  = test_all,
    norm_param               = norm_param,
    return_preds             = TRUE
  )

  preds[["crf"]] <- cesaro(
    train_data               = train_data,
    test_data                = test_data,
    rf_model                 = rf_model,
    rf_predictions_train_all = train_all,
    rf_predictions_test_all  = test_all,
    norm_param               = norm_param,
    return_preds             = TRUE
  )

  if (include_owrf) {
    preds[["owrf"]] <- tryCatch(
      owrf(
        x_train      = subset(train_data, select = -target),
        y_train      = train_data$target,
        x_test       = subset(test_data,  select = -target),
        y_test       = test_data$target,
        n_tree       = num_trees,
        norm_param   = norm_param,
        return_preds = TRUE
      ),
      error = function(e) rep(NA_real_, nrow(test_data))
    )
  }

  if (include_ridge) {
    preds[["ridge"]] <- ridge(
      rf_predictions_train_all = train_all,
      train_data               = train_data,
      rf_predictions_test_all  = test_all,
      test_data                = test_data,
      norm_param               = norm_param,
      return_preds             = TRUE
    )
  }

  if (include_ridge2nd) {
    n_half <- floor(n_train / 2)
    train1 <- train_data[seq_len(n_half), ]
    train2 <- train_data[(n_half + 1):n_train, ]
    rf2 <- ranger::ranger(
      target ~ .,
      data          = train1,
      num.trees     = num_trees,
      mtry          = floor((ncol(train1) - 1) / 3),
      replace       = TRUE,
      min.node.size = 5
    )
    X2     <- predict(rf2, train2, predict.all = TRUE)$predictions
    X_tst2 <- predict(rf2, test_data, predict.all = TRUE)$predictions
    raw    <- constrained_ridge(X2, train2$target, X_tst2)
    preds[["ridge_second"]] <- (raw * norm_param$sd) + norm_param$mean
  }

  # Compute MSE and MAE for each method
  calc <- function(p) c(mse = mse(p, y_test), mae = mae(p, y_test))
  metrics <- lapply(preds, function(p) {
    if (any(is.na(p))) c(mse = NA_real_, mae = NA_real_) else calc(p)
  })

  list(
    mse = sapply(metrics, function(m) m["mse"]),
    mae = sapply(metrics, function(m) m["mae"])
  )
}
