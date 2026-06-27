source("src/utils/get_mse.R")

library(CVXR)
library(ranger)

ols_second_sample <- function(train_data, test_data, num_trees, norm_param) {
  n <- nrow(train_data)
  n_half <- floor(n / 2)

  train1 <- train_data[1:n_half, ]
  train2 <- train_data[(n_half + 1):n, ]

  # Train RF on first half only
  rf_model <- ranger::ranger(
    target ~ .,
    data = train1,
    num.trees = num_trees,
    mtry = floor((ncol(train1) - 1) / 3),
    replace = TRUE,
    min.node.size = 5
  )

  # Tree predictions on second half (weight estimation sample)
  X_second <- predict(rf_model, train2, predict.all = TRUE)$predictions
  y_second <- train2$target

  # OLS with sum-to-one constraint (Timmermann 2006)
  w <- Variable(num_trees)
  objective <- sum_squares(y_second - X_second %*% w)
  constraints <- list(sum(w) == 1)
  prob <- Problem(Minimize(objective), constraints)
  solution <- solve(prob, num_iter = 100000, solver = "SCS")
  w_hat <- solution$getValue(w)

  # Predict on test data
  X_test <- predict(rf_model, test_data, predict.all = TRUE)$predictions
  preds <- X_test %*% w_hat
  preds <- (preds * norm_param$sd) + norm_param$mean

  mse(preds, test_data$target)
}
