# Purpose: Referee 2, Major Comment 3 (fixed feature dimension) and Minor
# Comment 4 (varying feature dimension) -- run the oracle-vs-feasible
# decomposition across a grid of training-set sizes and, optionally, a grid
# of total feature dimensions d (the Friedman-1 process always has 5
# relevant predictors, so d - 5 of them are pure noise).
library(parallel)
source("src/simulations/run_iteration_oracle.R")

get_simulation_oracle <- function(n_obs_grid,
                                   d_grid = 10,
                                   num_trees = 500,
                                   kappa = 2,
                                   noise_sd = 1,
                                   n_oracle = 50000,
                                   n_test = 20000,
                                   n_sim = 50,
                                   mc.cores = 1) {
  grid <- expand.grid(n_obs = n_obs_grid, d = d_grid)
  results <- lapply(seq_len(nrow(grid)), function(i) {
    n_obs <- grid$n_obs[i]
    d <- grid$d[i]
    reps <- mclapply(
      seq_len(n_sim),
      function(j) {
        run_iteration_oracle(
          n_obs = n_obs,
          num_trees = num_trees,
          kappa = kappa,
          d = d,
          noise_sd = noise_sd,
          n_oracle = n_oracle,
          n_test = n_test
        )
      },
      mc.cores = mc.cores
    )
    failed <- !vapply(reps, is.numeric, logical(1))
    if (any(failed)) {
      messages <- vapply(reps[failed], function(e) conditionMessage(e), character(1))
      stop("run_iteration_oracle failed for some repetitions: ", paste(messages, collapse = " | "))
    }
    as.data.frame(do.call(rbind, reps))
  })
  do.call(rbind, results)
}
