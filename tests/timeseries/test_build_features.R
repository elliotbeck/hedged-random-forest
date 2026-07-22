source("src/timeseries/build_features.R")

set.seed(1)
Tn <- 50
V <- 6
panel <- as.data.frame(matrix(rnorm(Tn * V), Tn, V))
colnames(panel) <- paste0("V", 1:V)
target_block <- rnorm(Tn)

n_lags <- 4
n_pcs <- 4
window_idx <- 20:40 # length 21, window_idx[1]-n_lags = 16 >= 1, OK

feat <- build_features(panel, target_block, window_idx, n_lags = n_lags, n_pcs = n_pcs)

stopifnot(nrow(feat) == length(window_idx))
stopifnot(ncol(feat) == n_lags * V + n_pcs + n_lags)
stopifnot(!any(is.na(feat)))

## spot-check one lag column by hand: V1_lag1 at window row k should equal
## panel$V1 at (window_idx[k] - 1)
stopifnot(all.equal(feat[["V1_lag1"]], panel$V1[window_idx - 1]))
stopifnot(all.equal(feat[["V3_lag4"]], panel$V3[window_idx - 4]))

## spot-check target AR lags
stopifnot(all.equal(feat[["target_lag1"]], target_block[window_idx - 1]))
stopifnot(all.equal(feat[["target_lag4"]], target_block[window_idx - 4]))

## PCs present and finite
stopifnot(all(c("PC1", "PC2", "PC3", "PC4") %in% colnames(feat)))
stopifnot(all(is.finite(feat$PC1)))

cat("ALL PASS\n")
