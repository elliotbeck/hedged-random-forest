source("src/simulations/get_simulation_owrf.R")
library(parallel)

# ── Parameters ───────────────────────────────────────────────────────────────
N_SIM     <- 1000     # set to 10 for quick experiment
NUM_TREES <- 500
KAPPAS    <- c(1, 2)
DATA_DIR  <- "data/owrf"
OUT_DIR   <- "results/owrf"

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Run all 12 datasets ───────────────────────────────────────────────────────
all_results <- mclapply(names(owrf_dataset_configs), function(nm) {
  get_simulation_owrf(
    dataset_name     = nm,
    cfg              = owrf_dataset_configs[[nm]],
    n_sim            = N_SIM,
    num_trees        = NUM_TREES,
    kappas           = KAPPAS,
    data_dir         = DATA_DIR,
    include_owrf     = TRUE,
    include_ridge    = TRUE,
    include_ridge2nd = TRUE
  )
}, mc.cores = 12L)
names(all_results) <- names(owrf_dataset_configs)

saveRDS(all_results, file.path(OUT_DIR, "owrf_results.rds"))

# ── Summary tables ────────────────────────────────────────────────────────────
build_table <- function(all_results, metric) {
  rows <- lapply(all_results, function(r) {
    if (is.null(r)) return(NULL)
    as.data.frame(as.list(r[[metric]]))
  })
  rows <- Filter(Negate(is.null), rows)
  tbl  <- do.call(rbind, rows)
  rownames(tbl) <- names(Filter(Negate(is.null), all_results))
  tbl
}

msfe_table <- build_table(all_results, "msfe")
mafe_table <- build_table(all_results, "mafe")

write.csv(msfe_table, file.path(OUT_DIR, "msfe_table.csv"))
write.csv(mafe_table, file.path(OUT_DIR, "mafe_table.csv"))

cat("\n=== MSFE table ===\n")
print(round(msfe_table, 4))
cat("\n=== MAFE table ===\n")
print(round(mafe_table, 4))
