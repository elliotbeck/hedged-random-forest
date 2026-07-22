# Purpose: Referee 1, Additional Comment 1 -- relate dataset characteristics
# (feature correlation, fat tails, dimensionality) to HRF's performance gains.
library(OpenML)
setOMLConfig(apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f")

datasets <- read.csv("metadata/numerical_regression.csv")

excess_kurtosis <- function(x) {
  x <- x[is.finite(x)]
  m <- mean(x)
  mean((x - m)^4) / (mean((x - m)^2)^2) - 3
}

get_characteristics <- function(dataset_id) {
  ds <- getOMLDataSet(data.id = dataset_id)
  data <- ds$data
  colnames(data)[colnames(data) == ds$target.features] <- "target"

  target <- data$target
  predictors <- data[, setdiff(colnames(data), "target"), drop = FALSE]
  is_num <- vapply(predictors, is.numeric, logical(1))
  predictors <- predictors[, is_num, drop = FALSE]

  cor_mat <- suppressWarnings(cor(predictors, use = "pairwise.complete.obs"))
  avg_abs_cor <- mean(abs(cor_mat[upper.tri(cor_mat)]), na.rm = TRUE)

  data.frame(
    dataset_id = dataset_id,
    n = nrow(data),
    p = ncol(predictors),
    avg_abs_cor = avg_abs_cor,
    kurtosis_target = excess_kurtosis(target)
  )
}

characteristics <- do.call(rbind, lapply(datasets$dataset_id, get_characteristics))
characteristics$dataset_name <- datasets$dataset_name[
  match(characteristics$dataset_id, datasets$dataset_id)
]
characteristics$n_over_p <- characteristics$n / characteristics$p

# Load simulation results and compute per-dataset performance-gain metrics
load_results_mean <- function(dataset_id) {
  results_mean <- get(load(paste0(
    "results/iid/weighted_rf_mean_", dataset_id, ".RData"
  )))
  results_mean$dataset_id <- dataset_id
  results_mean
}
results_mean <- do.call(rbind, lapply(datasets$dataset_id, load_results_mean))

get_gains <- function(dataset_id) {
  sub <- results_mean[results_mean$dataset_id == dataset_id, ]

  hrf_rf_gain <- function(row) sqrt(row$nls_2 / row$rf)
  kappa_gain <- function(row) sqrt(row$nls_2 / row$nls_1)

  # Average across all training-set sizes (200-5000) rather than just large
  # n: gains are largest at small n, so restricting to large n would discard
  # most of the variation we are trying to explain.
  data.frame(
    dataset_id = dataset_id,
    hrf_rf_gain_n5000 = hrf_rf_gain(sub[sub$n_obs == 5000, ]),
    hrf_rf_gain_avg = mean(vapply(seq_len(nrow(sub)), function(i) {
      hrf_rf_gain(sub[i, ])
    }, numeric(1))),
    kappa_gain_n5000 = kappa_gain(sub[sub$n_obs == 5000, ]),
    kappa_gain_avg = mean(vapply(seq_len(nrow(sub)), function(i) {
      kappa_gain(sub[i, ])
    }, numeric(1)))
  )
}
gains <- do.call(rbind, lapply(datasets$dataset_id, get_gains))

combined <- merge(characteristics, gains, by = "dataset_id")
combined <- combined[order(combined$dataset_name), ]

save(combined, file = "results/dataset_characteristics.RData")
write.csv(combined, "results/tables/dataset_characteristics.csv", row.names = FALSE)

cat("\n--- Dataset characteristics and performance gains ---\n")
print(combined[, c(
  "dataset_name", "n", "p", "n_over_p", "avg_abs_cor", "kurtosis_target",
  "hrf_rf_gain_avg", "kappa_gain_avg"
)], digits = 3)

cat("\n--- Spearman correlations with HRF/RF performance gain ---\n")
predictors_to_test <- c("avg_abs_cor", "kurtosis_target", "p", "n_over_p")
cor_table <- data.frame(characteristic = predictors_to_test)
cor_table$spearman_rho <- sapply(predictors_to_test, function(ch) {
  cor(combined[[ch]], combined$hrf_rf_gain_avg, method = "spearman")
})
print(cor_table, digits = 3)
