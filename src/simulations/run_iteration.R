run_iteration <- function(n_obs, data, num_trees, kappas,
                          include_wrf = TRUE,
                          include_crf = TRUE,
                          include_ridge = TRUE,
                          include_owrf = FALSE,
                          include_minvar = FALSE,
                          include_ols_second = FALSE) {
    # Shuffle rows
    data <- data[sample(seq_len(nrow(data))), ]

    # Data split
    train_data <- data[1:n_obs, ]
    test_data <- data[(n_obs + 1):nrow(data), ]

    # Standardizing target variable, not allowing for leakage
    norm_param <- list(
        mean = mean(train_data$target),
        sd = sd(train_data$target)
    )
    train_data$target <- (train_data$target - norm_param$mean) / norm_param$sd

    # Train default random forest
    rf_model <- ranger::ranger(
        target ~ .,
        data = train_data,
        num.trees = num_trees,
        mtry = floor((ncol(train_data) - 1) / 3),
        replace = TRUE,
        keep.inbag = TRUE,
        min.node.size = 5
    )

    # Tuned random forest predictions on test data
    rf_predictions <- predict(rf_model, test_data)$predictions
    rf_predictions <- (rf_predictions * norm_param$sd) + norm_param$mean

    # Random forest predictions and residuals on train data
    rf_predictions_train_all <- predict(
        rf_model,
        train_data,
        predict.all = TRUE
    )$predictions
    rf_residuals_train <- train_data$target - rf_predictions_train_all

    # Random forest predictions on test data
    rf_predictions_test_all <- predict(
        rf_model,
        test_data,
        predict.all = TRUE
    )$predictions

    # Calculate covariance matrix and mean vector of residuals
    mean_vector <- colMeans(rf_residuals_train)
    cov_matrix <- cov(rf_residuals_train)
    cov_matrix_shrinkage <- get_cov_qis(rf_residuals_train)

    # Get mse for all kappas and sample cov matrix, with tryCatch as sample cov matrix can be singular
    results_sample <- sapply(
        kappas,
        get_performance,
        mean_vector = mean_vector,
        cov_matrix = cov_matrix,
        predictions_test = rf_predictions_test_all,
        labels_test = test_data$target,
        mean = norm_param$mean,
        sd = norm_param$sd
    )
    names(results_sample) <- paste0("sample_", kappas)

    # Get mse for all kappas and nls cov matrix
    results_nls <- sapply(
        kappas,
        get_performance,
        mean_vector = mean_vector,
        cov_matrix = cov_matrix_shrinkage,
        predictions_test = rf_predictions_test_all,
        labels_test = test_data$target,
        mean = norm_param$mean,
        sd = norm_param$sd
    )
    names(results_nls) <- paste0("nls_", kappas)

    # Get winham et al. benchmark
    if (include_wrf) {
        mse_winham <- winham(
            train_data = train_data,
            test_data = test_data,
            rf_model = rf_model,
            rf_predictions_train_all = rf_predictions_train_all,
            rf_predictions_test_all = rf_predictions_test_all,
            norm_param = norm_param
        )
    }

    # Get cesaro benchmark
    if (include_crf) {
        mse_cesaro <- cesaro(
            train_data = train_data,
            test_data = test_data,
            rf_model = rf_model,
            rf_predictions_train_all = rf_predictions_train_all,
            rf_predictions_test_all = rf_predictions_test_all,
            norm_param = norm_param
        )
    }

    # Get ridge benchmark
    if (include_ridge) {
        mse_ridge <- ridge(
            rf_predictions_train_all = rf_predictions_train_all,
            train_data = train_data,
            rf_predictions_test_all = rf_predictions_test_all,
            test_data = test_data,
            norm_param = norm_param
        )
    }

    # Get owrf benchmark
    if (include_owrf) {
        mse_owrf <- owrf(
            x_train = subset(train_data, select = -target),
            y_train = train_data$target,
            x_test = subset(test_data, select = -target),
            y_test = test_data$target,
            n_tree = num_trees,
            norm_param = norm_param
        )
    }

    # Get minimum variance portfolio benchmarks (variance-only objective, bias-corrected)
    if (include_minvar) {
        results_minvar_sample <- sapply(
            kappas,
            minvar,
            mean_vector = mean_vector,
            cov_matrix = cov_matrix,
            predictions_test = rf_predictions_test_all,
            labels_test = test_data$target,
            mean = norm_param$mean,
            sd = norm_param$sd
        )
        names(results_minvar_sample) <- paste0("minvar_sample_", kappas)

        results_minvar_nls <- sapply(
            kappas,
            minvar,
            mean_vector = mean_vector,
            cov_matrix = cov_matrix_shrinkage,
            predictions_test = rf_predictions_test_all,
            labels_test = test_data$target,
            mean = norm_param$mean,
            sd = norm_param$sd
        )
        names(results_minvar_nls) <- paste0("minvar_nls_", kappas)
    }

    # Get second-sample ridge benchmark (Timmermann 2006)
    if (include_ols_second) {
        mse_ols_second <- get_ridge_second_sample(
            train_data = train_data,
            test_data = test_data,
            num_trees = num_trees,
            norm_param = norm_param
        )
    }

    # Bind results
    results <- c(
        n_obs = n_obs,
        rf = mse(rf_predictions, test_data$target),
        results_sample,
        results_nls
    )
    if (include_wrf)        results <- c(results, wrf        = mse_winham)
    if (include_crf)        results <- c(results, crf        = mse_cesaro)
    if (include_ridge)      results <- c(results, ridge      = mse_ridge)
    if (include_owrf)       results <- c(results, owrf       = mse_owrf)
    if (include_minvar)     results <- c(results, results_minvar_sample, results_minvar_nls)
    if (include_ols_second) results <- c(results, ridge_second = mse_ols_second)

    #  Store results
    return(results)
}
