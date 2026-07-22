source("src/timeseries/get_fredmd_panel.R")

## --- transform_series: exact formulas on a small synthetic vector ---
x <- c(10, 12, 15, 14, 18, 20)

y1 <- transform_series(x, 1)
stopifnot(identical(y1, x))

y2 <- transform_series(x, 2)
stopifnot(is.na(y2[1]))
stopifnot(all.equal(y2[2:6], diff(x)))

y3 <- transform_series(x, 3)
stopifnot(all(is.na(y3[1:2])))
stopifnot(all.equal(y3[3:6], diff(x, differences = 2)))

y4 <- transform_series(x, 4)
stopifnot(all.equal(y4, log(x)))

y5 <- transform_series(x, 5)
stopifnot(is.na(y5[1]))
stopifnot(all.equal(y5[2:6], diff(log(x))))

y6 <- transform_series(x, 6)
stopifnot(all(is.na(y6[1:2])))
stopifnot(all.equal(y6[3:6], diff(log(x), differences = 2)))

y7 <- transform_series(x, 7)
stopifnot(all(is.na(y7[1:2])))
pct <- x[2:6] / x[1:5] - 1
stopifnot(all.equal(y7[3:6], diff(pct)))

cat("transform_series: OK\n")

## --- interpolate_na: fills internal gaps, leaves leading/trailing NA alone ---
z <- c(NA, 10, NA, 20, 30, NA)
zi <- interpolate_na(z)
stopifnot(is.na(zi[1])) # leading NA: nothing to interpolate from
stopifnot(all.equal(zi[2:5], c(10, 15, 20, 30)))
stopifnot(is.na(zi[6])) # trailing NA: nothing to interpolate to
stopifnot(identical(interpolate_na(c(1, 2, 3)), c(1, 2, 3))) # no-op when no NA
stopifnot(identical(interpolate_na(rep(NA_real_, 5)), rep(NA_real_, 5))) # all-NA: no crash, unchanged
stopifnot(identical(interpolate_na(c(NA, 5, NA, NA)), c(NA, 5, NA, NA))) # single valid point: no crash, unchanged

cat("interpolate_na: OK\n")

## --- load_fredmd_panel: real data file ---
panel_data <- load_fredmd_panel("data/hrf-ts/2026-06-MD.csv")

stopifnot(inherits(panel_data$dates, "Date"))
stopifnot(min(panel_data$dates) == as.Date("1960-01-01"))
stopifnot(nrow(panel_data$raw) == length(panel_data$dates))
stopifnot(nrow(panel_data$transformed_full) == length(panel_data$dates))
stopifnot(ncol(panel_data$panel) <= ncol(panel_data$transformed_full))
stopifnot(all(colSums(is.na(panel_data$panel)) == 0))
# regression guard: transforms must be computed on full history BEFORE restricting
# to start_date, or every differenced column gets a leading NA and panel collapses
# to only the already-stationary (tcode 1/4) columns (~20 instead of ~110+)
stopifnot(ncol(panel_data$panel) > 100)
stopifnot("CPIAUCSL" %in% names(panel_data$tcodes))
stopifnot(panel_data$tcodes[["CPIAUCSL"]] == 6)
stopifnot(panel_data$tcodes[["FEDFUNDS"]] == 2)
stopifnot(panel_data$tcodes[["BAAFFM"]] == 1)

# CPIAUCSL is missing exactly 2025-10-01 in this vintage; interpolation must have filled it
cpi_raw <- panel_data$raw[["CPIAUCSL"]]
stopifnot(!any(is.na(cpi_raw)))
# CPIAUCSL raw levels should look like an index (order of 10s-100s), not a small pct change
stopifnot(all(cpi_raw > 10 & cpi_raw < 1000))

cat("load_fredmd_panel: OK\n")
cat("ALL PASS\n")
