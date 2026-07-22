# Purpose: Referee 2, Major Comment 3 -- run the oracle-vs-feasible
# decomposition across a grid of (realistic) training-set sizes and
# repetitions.
library(parallel)
source("src/simulations/run_iteration_oracle.R")

get_simulation_oracle <- function(n_obs_grid,
                                   num_trees = 500,
                                   kappa = 2,
                                   d = 10,
                                   noise_sd = 1,
                                   n_oracle = 50000,
                                   n_test = 20000,
                                   n_sim = 50,
                                   mc.cores = 1) {
  results <- lapply(n_obs_grid, function(n_obs) {
    reps <- mclapply(
      seq_len(n_sim),
      function(i) {
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
    do.call(rbind, reps)
  })
  as.data.frame(do.call(rbind, results))
}
