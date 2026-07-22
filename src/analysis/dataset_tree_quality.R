# Purpose: Referee 1, Additional Comment 1 -- compute, per dataset, the
# dispersion in individual-tree accuracy (some trees may simply be worse
# than others), as a candidate explanation for HRF's gains: HRF can
# downweight persistently worse-performing trees. Tree accuracy is measured
# out-of-bag (tPE_j, eq. win0), the same construction used for the WRF
# benchmark in get_winham_benchmark.R, rather than in-sample: in-sample
# error is a biased measure of tree quality since deep trees can nearly
# fit their own bootstrap sample.
library(OpenML)
library(ranger)
library(parallel)
setOMLConfig(apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f")

datasets <- read.csv("metadata/numerical_regression.csv")
set.seed(42)

get_tree_quality <- function(dataset_id, n_obs = 5000, num_trees = 500) {
  ds <- getOMLDataSet(data.id = dataset_id)
  data <- ds$data
  colnames(data)[colnames(data) == ds$target.features] <- "target"
  data <- data[sample(seq_len(nrow(data)), min(nrow(data), 10000)), ]

  train_data <- data[seq_len(min(n_obs, nrow(data))), ]
  train_data$target <- as.numeric(scale(train_data$target))

  rf_model <- ranger::ranger(
    target ~ .,
    data = train_data,
    num.trees = num_trees,
    mtry = floor((ncol(train_data) - 1) / 3),
    replace = TRUE,
    keep.inbag = TRUE,
    min.node.size = 5
  )

  predictions_train_all <- predict(
    rf_model, train_data, predict.all = TRUE
  )$predictions

  # Mask in-bag predictions so only each tree's out-of-bag predictions
  # remain, exactly as in get_winham_benchmark.R.
  inbag_counts <- do.call(cbind, rf_model$inbag.counts)
  predictions_train_all[inbag_counts > 0] <- NA

  # Per-tree out-of-bag accuracy tPE_j (eq. win0): mean absolute OOB error.
  tree_tpe <- colMeans(
    abs(predictions_train_all - train_data$target),
    na.rm = TRUE
  )

  # Dispersion in individual-tree accuracy: some trees may simply be worse
  # (higher tPE) than others. Measured as the coefficient of variation
  # (scale-free) of the p = 500 individual out-of-bag tree tPE_j.
  tree_quality_cv <- sd(tree_tpe) / mean(tree_tpe)

  data.frame(
    dataset_id = dataset_id,
    tree_quality_cv = tree_quality_cv
  )
}

get_tree_quality_retry <- function(dataset_id, max_tries = 3) {
  for (attempt in seq_len(max_tries)) {
    result <- tryCatch(get_tree_quality(dataset_id), error = function(e) e)
    if (!inherits(result, "error")) {
      return(result)
    }
    message(sprintf(
      "Dataset %s failed (attempt %d/%d): %s",
      dataset_id, attempt, max_tries, conditionMessage(result)
    ))
  }
  stop(sprintf("Dataset %s failed after %d attempts", dataset_id, max_tries))
}

tree_quality <- mclapply(
  datasets$dataset_id,
  get_tree_quality_retry,
  mc.cores = nrow(datasets)
)
failed <- !vapply(tree_quality, is.data.frame, logical(1))
if (any(failed)) {
  stop(
    "get_tree_quality_retry did not return a data.frame for dataset_id(s): ",
    paste(datasets$dataset_id[failed], collapse = ", ")
  )
}
tree_quality <- do.call(rbind, tree_quality)
stopifnot(nrow(tree_quality) == nrow(datasets))

save(tree_quality, file = "results/dataset_tree_quality.RData")

# Merge with characteristics/gains computed in dataset_characteristics_analysis.R
load("results/dataset_characteristics.RData")
n_before <- nrow(combined)
combined <- combined[, setdiff(colnames(combined), c("avg_abs_tree_cor", "tree_quality_cv"))]
combined <- merge(combined, tree_quality, by = "dataset_id")
stopifnot(nrow(combined) == n_before)
save(combined, file = "results/dataset_characteristics.RData")
write.csv(combined, "results/tables/dataset_characteristics.csv", row.names = FALSE)

cat("\n--- Tree-quality dispersion added ---\n")
print(combined[, c("dataset_name", "avg_abs_cor", "tree_quality_cv")], digits = 3)

cat("\n--- Spearman correlations with HRF/RF performance gain ---\n")
predictors_to_test <- c("avg_abs_cor", "tree_quality_cv", "kurtosis_target", "p", "n_over_p")
cor_table <- data.frame(characteristic = predictors_to_test)
cor_table$spearman_rho <- sapply(predictors_to_test, function(ch) {
  cor(combined[[ch]], combined$hrf_rf_gain_avg, method = "spearman")
})
print(cor_table, digits = 3)
