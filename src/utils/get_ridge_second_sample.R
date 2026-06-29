source("src/utils/get_mse.R")
source("src/utils/get_ridge_benchmark.R")

library(ranger)

get_ridge_second_sample <- function(train_data, test_data, num_trees, norm_param,
                                min_node_size = 5) {
  n      <- nrow(train_data)
  n_half <- floor(n / 2)

  train1 <- train_data[1:n_half, ]
  train2 <- train_data[(n_half + 1):n, ]

  rf_model <- ranger::ranger(
    target ~ .,
    data          = train1,
    num.trees     = num_trees,
    mtry          = floor((ncol(train1) - 1) / 3),
    replace       = TRUE,
    min.node.size = min_node_size
  )

  X2     <- predict(rf_model, train2, predict.all = TRUE)$predictions
  X_test <- predict(rf_model, test_data, predict.all = TRUE)$predictions

  preds <- constrained_ridge(
    X_train = X2,
    y_train = train2$target,
    X_test  = X_test
  )
  preds <- (preds * norm_param$sd) + norm_param$mean
  mse(preds, test_data$target)
}
