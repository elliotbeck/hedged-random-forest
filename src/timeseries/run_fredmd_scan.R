library(parallel)

run_fredmd_scan <- function(fredmd_data, targets, horizons, oos_start_date,
                             mc_cores = 30, window_length = 360, ...) {
  dates <- fredmd_data$dates
  panel <- fredmd_data$panel
  Tn <- length(dates)

  oos_idx <- which(dates >= oos_start_date)
  min_idx <- window_length + 4 + 1 # room for the rolling window plus lag history
  oos_idx <- oos_idx[oos_idx >= min_idx]

  ## flatten the (target, origin) grid for even load balancing across cores
  jobs <- expand.grid(target_row = seq_len(nrow(targets)), origin_idx = oos_idx)

  run_one <- function(k) {
    target_row <- jobs$target_row[k]
    origin_idx <- jobs$origin_idx[k]
    id <- targets$id[target_row]
    tcode <- targets$tcode[target_row]

    target_raw <- fredmd_data$raw[[id]]
    target_block <- get_target_building_block(target_raw, tcode)

    result <- get_forecast_vintage_ts(
      panel, target_block, target_raw, origin_idx, horizons, tcode,
      window_length = window_length, ...
    )
    result$id <- id
    result$origin_date <- dates[origin_idx]
    result
  }

  results <- mclapply(seq_len(nrow(jobs)), run_one, mc.cores = mc_cores)
  do.call(rbind, results)
}
