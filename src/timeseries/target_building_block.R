get_target_building_block <- function(x, tcode) {
  if (tcode == 1) {
    x
  } else if (tcode == 2) {
    transform_series(x, 2)
  } else if (tcode %in% c(5, 6)) {
    transform_series(x, 5)
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
