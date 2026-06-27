source("src/utils/get_mse.R")

library(CVXR)

get_minvar_weights <- function(cov_matrix, kappa) {
  w <- Variable(ncol(cov_matrix))
  objective <- quad_form(w, cov_matrix)
  constraints <- list(
    sum(w) == 1,
    sum(abs(w)) <= kappa
  )
  prob <- Problem(Minimize(objective), constraints)
  solution <- solve(prob, num_iter = 100000, solver = "SCS")
  return(solution$getValue(w))
}

minvar <- function(mean_vector, cov_matrix, kappa, predictions_test, labels_test, mean, sd) {
  w <- get_minvar_weights(cov_matrix, kappa)

  # Bias-corrected prediction: add average training residual w'mu to correct
  # for the systematic bias that the variance-only objective ignores
  bias_correction <- as.numeric(t(w) %*% mean_vector)
  preds <- as.matrix(predictions_test) %*% w + bias_correction
  preds <- (preds * sd) + mean

  mse(preds, labels_test)
}
