# Purpose: Referee 2, Minor Comment 4 -- reuse the Major Comment 3
# oracle-vs-feasible decomposition (same Friedman-1 data-generating process,
# same training-set-size grid), but now also vary the number of irrelevant
# (pure-noise) predictors, to see whether HRF's advantage over RF holds up
# as an increasing fraction of the predictors carries no signal.
library(parallel)
source("src/simulations/run_iteration_oracle.R")

get_simulation_oracle_dim <- function(n_obs_grid,
                                       d_irrelevant_grid,
                                       num_trees = 500,
                                       kappa = 2,
                                       noise_sd = 1,
                                       n_oracle = 50000,
                                       n_test = 20000,
                                       n_sim = 50,
                                       mc.cores = 1) {
  grid <- expand.grid(n_obs = n_obs_grid, d_irrelevant = d_irrelevant_grid)
  results <- lapply(seq_len(nrow(grid)), function(i) {
    n_obs <- grid$n_obs[i]
    d_irrelevant <- grid$d_irrelevant[i]
    reps <- mclapply(
      seq_len(n_sim),
      function(j) {
        run_iteration_oracle(
          n_obs = n_obs,
          num_trees = num_trees,
          kappa = kappa,
          d = 5 + d_irrelevant,
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
    out <- as.data.frame(do.call(rbind, reps))
    out$d_irrelevant <- d_irrelevant
    out
  })
  do.call(rbind, results)
}
