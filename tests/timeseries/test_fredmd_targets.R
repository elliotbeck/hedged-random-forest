source("src/timeseries/fredmd_targets.R")

stopifnot(is.data.frame(fredmd_targets))
stopifnot(nrow(fredmd_targets) == 41)
stopifnot(all(c("id", "group", "tcode") %in% colnames(fredmd_targets)))
stopifnot(!anyDuplicated(fredmd_targets$id))
stopifnot(all(fredmd_targets$tcode %in% c(1, 2, 5, 6)))
stopifnot(sum(fredmd_targets$group == "Prices") == 20)
stopifnot(sum(fredmd_targets$group == "Rate") == 9)
stopifnot(sum(fredmd_targets$group == "Spread") == 8)
stopifnot(sum(fredmd_targets$group == "FX") == 4)
stopifnot(all(fredmd_targets$tcode[fredmd_targets$group == "Prices"] == 6))
stopifnot(all(fredmd_targets$tcode[fredmd_targets$group == "Rate"] == 2))
stopifnot(all(fredmd_targets$tcode[fredmd_targets$group == "Spread"] == 1))
stopifnot(all(fredmd_targets$tcode[fredmd_targets$group == "FX"] == 5))

## every listed id must actually exist as a column in the real data file
panel_cols <- colnames(read.csv("data/hrf-ts/2026-06-MD.csv", check.names = FALSE))
stopifnot(all(fredmd_targets$id %in% panel_cols))

cat("ALL PASS\n")
