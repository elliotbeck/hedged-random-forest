source("src/timeseries/get_fredmd_panel.R")
source("src/timeseries/target_building_block.R")

x <- c(100, 102, 101, 105, 108, 110)

## tcode 1: building block is the identity
stopifnot(identical(get_target_building_block(x, 1), x))

## tcode 2: building block is the first difference
b2 <- get_target_building_block(x, 2)
stopifnot(all.equal(b2, transform_series(x, 2)))

## tcode 5 or 6: building block is Delta-log (one order lower than tcode 6)
b5 <- get_target_building_block(x, 5)
b6 <- get_target_building_block(x, 6)
stopifnot(all.equal(b5, transform_series(x, 5)))
stopifnot(identical(b5, b6))

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

cat("ALL PASS\n")
