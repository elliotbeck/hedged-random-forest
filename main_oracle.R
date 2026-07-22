# Purpose: Referee 2, Major Comment 3 -- decompose HRF(Sigma-hat) - RF into
# a structural-advantage term (HRF(Sigma) - RF, using oracle moments for the
# SAME p trees, estimated on a large separate out-of-sample set) and an
# estimation-effect term (HRF(Sigma-hat) - HRF(Sigma)), for Sigma-hat
# estimated in-sample at a realistic training-set size, with both the plain
# sample covariance and the NLS/QIS shrinkage estimator.
source("src/simulations/get_simulation_oracle.R")

set.seed(42)
n_obs_grid <- c(200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000)
n_trees <- 500
kappa <- 2
n_sim <- 100
n_oracle <- 50000
n_test <- 20000

results_oracle <- get_simulation_oracle(
  n_obs_grid = n_obs_grid,
  num_trees = n_trees,
  kappa = kappa,
  d = 10,
  noise_sd = 1,
  n_oracle = n_oracle,
  n_test = n_test,
  n_sim = n_sim,
  mc.cores = min(n_sim, parallel::detectCores())
)

dir.create("results/oracle", showWarnings = FALSE, recursive = TRUE)
save(results_oracle, file = "results/oracle/results_oracle.RData")
