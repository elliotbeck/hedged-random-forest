source("src/timeseries/get_fredmd_panel.R")

pct_change <- function(x) {
  n <- length(x)
  y <- rep(NA_real_, n)
  y[2:n] <- x[2:n] / x[1:(n - 1)] - 1
  y
}

get_target_building_block <- function(x, tcode) {
  if (tcode == 1) {
    x
  } else if (tcode == 2) {
    transform_series(x, 2)
  } else if (tcode %in% c(5, 6)) {
    pct_change(x)
  } else {
    stop("Unsupported tcode for building block: ", tcode)
  }
}

aggregate_building_block <- function(x_t, block_forecasts, tcode) {
  if (tcode == 1) {
    block_forecasts
  } else if (tcode == 2) {
    x_t + cumsum(block_forecasts)
  } else if (tcode %in% c(5, 6)) {
    x_t * cumprod(1 + block_forecasts)
  } else {
    stop("Unsupported tcode for aggregation: ", tcode)
  }
}
