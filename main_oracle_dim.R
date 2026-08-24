# Purpose: Referee 2, Minor Comment 4 -- assess HRF vs. RF as the number of
# irrelevant (pure-noise) predictors grows, holding the same Friedman-1
# relevant structure (x1-x5) and the same training-set-size grid as Major
# Comment 3 fixed.
source("src/simulations/get_simulation_oracle_dim.R")

set.seed(42)
n_obs_grid <- c(200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000)
d_irrelevant_grid <- c(10, 50, 100, 250, 500)
n_trees <- 500
kappa <- 2
n_sim <- 100
n_oracle <- 50000
n_test <- 20000

results_oracle_dim <- get_simulation_oracle_dim(
  n_obs_grid = n_obs_grid,
  d_irrelevant_grid = d_irrelevant_grid,
  num_trees = n_trees,
  kappa = kappa,
  noise_sd = 1,
  n_oracle = n_oracle,
  n_test = n_test,
  n_sim = n_sim,
  mc.cores = min(n_sim, parallel::detectCores())
)

dir.create("results/oracle", showWarnings = FALSE, recursive = TRUE)
save(results_oracle_dim, file = "results/oracle/results_oracle_dim.RData")
