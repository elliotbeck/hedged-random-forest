source("src/utils/get_weights.R")
source("src/timeseries/get_fredmd_panel.R")
source("src/timeseries/target_building_block.R")
source("src/cov_estimators/get_cov_ewma_shrink.R")
source("src/timeseries/build_features.R")
source("src/timeseries/get_forecast_lead.R")
source("src/timeseries/get_forecast_vintage_ts.R")

set.seed(3)
Tn <- 420
V <- 10
panel <- as.data.frame(matrix(rnorm(Tn * V), Tn, V))
colnames(panel) <- paste0("V", 1:V)

## Construct a target whose raw level follows a random walk with drift, so both
## tcode=2 (level, additive building block) and tcode=6 (price-like, multiplicative
## building block) behave sensibly with strictly-positive levels.
target_raw <- cumsum(rnorm(Tn, mean = 0.05, sd = 0.3)) + 100
stopifnot(all(target_raw > 0))

origin_idx <- 400
window_length <- 360
horizons <- 1:3

for (tcode in c(1, 2, 6)) {
  target_block <- get_target_building_block(target_raw, tcode)
  result <- get_forecast_vintage_ts(
    panel, target_block, target_raw, origin_idx, horizons, tcode,
    window_length = window_length, num_trees = 50
  )
  stopifnot(nrow(result) == length(horizons))
  stopifnot(all(c("horizon", "rf_forecast", "hrf_forecast", "actual") %in% colnames(result)))
  stopifnot(identical(result$horizon, horizons))
  stopifnot(all(is.finite(result$rf_forecast)))
  stopifnot(all(is.finite(result$hrf_forecast)))
  ## actual should match target_raw[origin_idx + horizon] exactly
  stopifnot(all.equal(result$actual, target_raw[origin_idx + horizons]))
  cat("tcode", tcode, "OK\n")
}

## out-of-range horizon yields NA actual, not an error
result_far <- get_forecast_vintage_ts(
  panel, get_target_building_block(target_raw, 2), target_raw, origin_idx, c(1, 30), 2,
  window_length = window_length, num_trees = 50
)
stopifnot(is.na(result_far$actual[2]))

cat("ALL PASS\n")
