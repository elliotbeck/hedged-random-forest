get_forecast_vintage_ts <- function(panel, target_block, target_raw, origin_idx, horizons, tcode,
                                     window_length = 360, n_lags = 4, n_pcs = 4,
                                     num_trees = 500, kappa = 2,
                                     ewma_lambda = 0.15, ewma_H = 6,
                                     winsorize_prob = 0.01) {
  window_idx <- (origin_idx - window_length + 1):origin_idx
  features <- build_features(panel, target_block, window_idx, n_lags = n_lags, n_pcs = n_pcs)
  W <- nrow(features)

  max_h <- max(horizons)
  block_rf <- numeric(max_h)
  block_hrf <- numeric(max_h)

  for (j in seq_len(max_h)) {
    n_train <- W - j
    features_train <- features[seq_len(n_train), , drop = FALSE]
    label_train <- target_block[window_idx[(1 + j):W]]

    if (winsorize_prob > 0) {
      qs <- quantile(label_train, probs = c(winsorize_prob, 1 - winsorize_prob), na.rm = TRUE)
      label_train <- pmin(pmax(label_train, qs[1]), qs[2])
    }

    features_test <- features[W, , drop = FALSE]

    lead_result <- get_forecast_lead(features_train, label_train, features_test,
      num_trees = num_trees, kappa = kappa, ewma_lambda = ewma_lambda, ewma_H = ewma_H
    )
    block_rf[j] <- lead_result$rf
    block_hrf[j] <- lead_result$hrf
  }

  x_t <- target_raw[origin_idx]
  raw_rf <- aggregate_building_block(x_t, block_rf, tcode)
  raw_hrf <- aggregate_building_block(x_t, block_hrf, tcode)

  Tn <- length(target_raw)
  data.frame(
    horizon = horizons,
    rf_forecast = raw_rf[horizons],
    hrf_forecast = raw_hrf[horizons],
    actual = ifelse(origin_idx + horizons <= Tn, target_raw[pmin(origin_idx + horizons, Tn)], NA_real_)
  )
}
