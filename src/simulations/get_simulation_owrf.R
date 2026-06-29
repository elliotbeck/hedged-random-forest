source("src/utils/get_mse.R")
source("src/utils/get_mae.R")
source("src/cov_estimators/get_cov_qis.R")
source("src/utils/get_performance.R")   # sources get_weights.R internally
source("src/utils/get_winham_benchmark.R")
source("src/utils/get_cesaro_benchmark.R")
source("src/utils/get_owrf_benchmark.R")
source("src/utils/get_ridge_benchmark.R")
source("src/utils/get_ridge_second_sample.R")
source("src/simulations/run_iteration_owrf.R")

library(ranger)
library(randomForest)
library(CVXR)

# Dataset configurations for the 12 Chen et al. datasets.
# Each entry: list(file, sep, header, target_col, drop_cols)
# drop_cols: character vector of column names to exclude (besides target)
owrf_dataset_configs <- list(
  ASN     = list(file="ASN.csv",     sep="\t", header=FALSE, target_col="V6",       drop_cols=character(0)),
  CCPP    = list(file="CCPP.csv",    sep=",",  header=FALSE, target_col="V5",        drop_cols=character(0)),
  CCS     = list(file="CCS.csv",     sep=",",  header=TRUE,  target_col="Concrete.compressive.strength.MPa..megapascals..", drop_cols=character(0)),
  CST     = list(file="CST.csv",     sep=",",  header=FALSE, target_col="V10",       drop_cols=character(0)),
  EE      = list(file="EE.csv",      sep=",",  header=FALSE, target_col="V9",        drop_cols="V10"),
  PT      = list(file="PT.csv",      sep=",",  header=TRUE,  target_col="total_UPDRS", drop_cols=c("subject.", "motor_UPDRS")),
  QSAR    = list(file="QSAR.csv",    sep=";",  header=FALSE, target_col="V9",        drop_cols=character(0)),
  SM      = list(file="SM.csv",      sep=",",  header=FALSE, target_col="V5",        drop_cols=character(0)),
  YH      = list(file="YH.csv",      sep="",   header=FALSE, target_col="V7",        drop_cols=character(0)),
  housing = list(file="housing.csv", sep="",   header=FALSE, target_col="V14",       drop_cols=character(0)),
  servo   = list(file="servo.csv",   sep=",",  header=TRUE,  target_col="class",     drop_cols=character(0)),
  tecator = list(file="tecator.csv", sep=",",  header=TRUE,  target_col="fat",
                 drop_cols=c("moisture","protein",
                             paste0("principal_component_", 1:22)))
)

# Load and preprocess one dataset. Returns data.frame with column "target".
load_owrf_dataset <- function(cfg, data_dir = "data/owrf") {
  # sep="" means whitespace-delimited (read.table behaviour); read.csv uses it literally
  if (cfg$sep == "") {
    df <- read.table(file.path(data_dir, cfg$file), header = cfg$header,
                     stringsAsFactors = FALSE)
  } else {
    df <- read.csv(file.path(data_dir, cfg$file), sep = cfg$sep,
                   header = cfg$header, stringsAsFactors = FALSE)
  }

  # Drop excluded columns
  if (length(cfg$drop_cols) > 0) {
    df <- df[, !(colnames(df) %in% cfg$drop_cols), drop = FALSE]
  }

  # Rename target
  tc <- cfg$target_col
  if (!(tc %in% colnames(df))) stop("Target column not found: ", tc)
  colnames(df)[colnames(df) == tc] <- "target"
  df$target <- as.numeric(df$target)

  # Encode categorical features numerically
  has_char <- any(sapply(df[, colnames(df) != "target", drop = FALSE], is.character))
  if (has_char) {
    mm <- model.matrix(target ~ ., data = df)[, -1, drop = FALSE]
    df <- cbind(as.data.frame(mm), target = df$target)
  } else {
    for (col in setdiff(colnames(df), "target")) df[[col]] <- as.numeric(df[[col]])
  }

  df[complete.cases(df), ]
}

# Run n_sim iterations for one dataset and return averaged MSFE/MAFE.
get_simulation_owrf <- function(dataset_name, cfg, n_sim = 1000,
                                num_trees = 500, kappas = c(1, 2),
                                data_dir  = "data/owrf",
                                include_owrf     = TRUE,
                                include_ridge    = TRUE,
                                include_ridge2nd = TRUE) {
  cat("Dataset:", dataset_name, "\n")
  data <- load_owrf_dataset(cfg, data_dir)
  cat("  n =", nrow(data), "| p =", ncol(data) - 1, "\n")

  results <- vector("list", n_sim)
  for (b in seq_len(n_sim)) {
    results[[b]] <- tryCatch(
      run_iteration_owrf(
        data             = data,
        num_trees        = num_trees,
        kappas           = kappas,
        include_owrf     = include_owrf,
        include_ridge    = include_ridge,
        include_ridge2nd = include_ridge2nd
      ),
      error = function(e) {
        message("  Iteration ", b, " failed: ", conditionMessage(e))
        NULL
      }
    )
  }

  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) {
    warning("All iterations failed for dataset: ", dataset_name)
    return(NULL)
  }

  mse_mat <- do.call(rbind, lapply(results, function(r) r$mse))
  mae_mat <- do.call(rbind, lapply(results, function(r) r$mae))

  list(
    dataset = dataset_name,
    n       = nrow(data),
    p       = ncol(data) - 1,
    n_iter  = length(results),
    msfe    = colMeans(mse_mat, na.rm = TRUE),
    mafe    = colMeans(mae_mat, na.rm = TRUE)
  )
}
