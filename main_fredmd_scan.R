source("src/utils/get_weights.R")
source("src/timeseries/get_fredmd_panel.R")
source("src/timeseries/target_building_block.R")
source("src/cov_estimators/get_cov_ewma_shrink.R")
source("src/timeseries/build_features.R")
source("src/timeseries/get_forecast_lead.R")
source("src/timeseries/get_forecast_vintage_ts.R")
source("src/timeseries/fredmd_targets.R")
source("src/timeseries/run_fredmd_scan.R")

fredmd_data <- load_fredmd_panel("data/hrf-ts/2026-06-MD.csv", start_date = as.Date("1960-01-01"))

dir.create("results/fredmd", recursive = TRUE, showWarnings = FALSE)

# --- Phase 1: h = 1 scan across all 41 targets ---
phase1 <- run_fredmd_scan(
  fredmd_data, fredmd_targets, horizons = 1,
  oos_start_date = as.Date("1990-01-01"), mc_cores = 30
)
saveRDS(phase1, "results/fredmd/phase1_h1_scan.rds")

rmse_ratio <- function(df) {
  complete <- df[!is.na(df$actual), ]
  rmse_rf <- sqrt(mean((complete$actual - complete$rf_forecast)^2))
  rmse_hrf <- sqrt(mean((complete$actual - complete$hrf_forecast)^2))
  mae_rf <- mean(abs(complete$actual - complete$rf_forecast))
  mae_hrf <- mean(abs(complete$actual - complete$hrf_forecast))
  data.frame(rmse_ratio = rmse_hrf / rmse_rf, mae_ratio = mae_hrf / mae_rf, n_obs = nrow(complete))
}

phase1_summary <- do.call(rbind, lapply(split(phase1, phase1$id), rmse_ratio))
phase1_summary$id <- rownames(phase1_summary)
phase1_summary <- merge(phase1_summary, fredmd_targets, by = "id")
phase1_summary <- phase1_summary[order(phase1_summary$group, phase1_summary$rmse_ratio), ]

write.csv(phase1_summary, "results/fredmd/phase1_h1_summary.csv", row.names = FALSE)
print(phase1_summary)
