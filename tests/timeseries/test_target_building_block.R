source("src/timeseries/get_fredmd_panel.R")
source("src/timeseries/target_building_block.R")

x <- c(100, 102, 101, 105, 108, 110)

## tcode 1: building block is the identity
stopifnot(identical(get_target_building_block(x, 1), x))

## tcode 2: building block is the first difference
b2 <- get_target_building_block(x, 2)
stopifnot(all.equal(b2, transform_series(x, 2)))

## tcode 5 or 6: building block is the plain percent change (NOT Delta-log),
## one order lower than tcode 6, matching the companion paper's eq. 4.3 exactly
b5 <- get_target_building_block(x, 5)
b6 <- get_target_building_block(x, 6)
expected_pct <- c(NA, x[2:6] / x[1:5] - 1)
stopifnot(all.equal(b5, expected_pct))
stopifnot(identical(b5, b6))
stopifnot(all.equal(pct_change(x), expected_pct))

## aggregate_building_block: tcode 1, forecasts ARE the raw-level forecasts
agg1 <- aggregate_building_block(x_t = 100, block_forecasts = c(101, 103, 99), tcode = 1)
stopifnot(all.equal(agg1, c(101, 103, 99)))

## aggregate_building_block: tcode 2, cumulative sum on top of x_t
agg2 <- aggregate_building_block(x_t = 100, block_forecasts = c(1, -0.5, 2), tcode = 2)
stopifnot(all.equal(agg2, c(101, 100.5, 102.5)))

## aggregate_building_block: tcode 5/6, cumulative product on top of x_t
agg5 <- aggregate_building_block(x_t = 100, block_forecasts = c(0.01, -0.02, 0.03), tcode = 5)
expected5 <- 100 * cumprod(1 + c(0.01, -0.02, 0.03))
stopifnot(all.equal(agg5, expected5))

## round-trip: aggregating a real series' own building blocks must exactly
## reproduce its subsequent raw levels -- this is an exact identity for the
## percent-change building block (x_t/x_{t-1}-1) and would fail if the
## building block were instead Delta-log(x) (only a first-order approximation
## under cumulative-product aggregation)
bb <- get_target_building_block(x, 5)
agg_path <- aggregate_building_block(x_t = x[1], block_forecasts = bb[2:6], tcode = 5)
stopifnot(all.equal(agg_path, x[2:6]))

cat("ALL PASS\n")
