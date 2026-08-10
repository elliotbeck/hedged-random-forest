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

results <- run_fredmd_scan(
  fredmd_data, fredmd_targets, horizons = 1,
  oos_start_date = as.Date("1990-01-01"), mc_cores = 30
)
saveRDS(results, "results/fredmd/phase1_h1_scan.rds")

rmse_ratio <- function(df) {
  complete <- df[!is.na(df$actual), ]
  rmse_rf <- sqrt(mean((complete$actual - complete$rf_forecast)^2))
  rmse_hrf <- sqrt(mean((complete$actual - complete$hrf_forecast)^2))
  rmse_hrf_minvar <- sqrt(mean((complete$actual - complete$hrf_minvar_forecast)^2))
  mae_rf <- mean(abs(complete$actual - complete$rf_forecast))
  mae_hrf <- mean(abs(complete$actual - complete$hrf_forecast))
  mae_hrf_minvar <- mean(abs(complete$actual - complete$hrf_minvar_forecast))
  data.frame(
    rmse_ratio = rmse_hrf / rmse_rf, mae_ratio = mae_hrf / mae_rf,
    rmse_ratio_minvar = rmse_hrf_minvar / rmse_rf, mae_ratio_minvar = mae_hrf_minvar / mae_rf,
    n_obs = nrow(complete)
  )
}

results_summary <- do.call(rbind, lapply(split(results, results$id), rmse_ratio))
results_summary$id <- rownames(results_summary)
results_summary <- merge(results_summary, fredmd_targets, by = "id")
results_summary <- results_summary[order(results_summary$group, results_summary$rmse_ratio), ]

write.csv(results_summary, "results/fredmd/results_summary.csv", row.names = FALSE)
print(results_summary)
