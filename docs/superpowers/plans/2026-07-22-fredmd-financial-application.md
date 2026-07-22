# FRED-MD Financial-Econometric Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rolling-window HRF-vs-RF backtest pipeline for 41 FRED-MD financial series (Prices + Interest & Exchange Rates groups), run the Phase-1 (h=1) scan across all of them, and produce an RMSE/MAE ratio results table.

**Architecture:** New `src/timeseries/` module (FRED-MD loading/transforms, feature engineering, per-lead RF+HRF forecasting, vintage/horizon aggregation) plus a new `get_cov_ewma_shrink.R` estimator in `src/cov_estimators/`, orchestrated by `main_fredmd_scan.R`. Reuses the existing `src/utils/get_weights.R` HRF weight solver unchanged.

**Tech Stack:** R, `ranger` (RF), `CVXR` (HRF weight QP, already a project dependency), base R only for everything else (no new package dependencies — in particular, no `BVAR`, which is referenced by the existing-but-unused `src/utils/get_fredmd_data.R` and is not in `renv.lock`).

## Global Constraints

- Data source: `data/hrf-ts/2026-06-MD.csv` (FRED-MD vintage, Jan 1959–May 2026, raw untransformed levels + a `Transform:` tcode row). No network/API calls.
- Estimation start date: 1960-01-01 (matches the companion paper; avoids a single stray early-1959 NA present in several series).
- Rolling window: 360 months. OOS period for Phase 1: nominally Jan 1990 → latest available month, but `run_fredmd_scan.R`'s `min_idx <- window_length + 4 + 1` buffer (room for the rolling window plus 4 lags) pushes the first *usable* origin to **1990-05**, not 1990-01. Realized Phase-1 run: origins 1990-05 through 2026-04 inclusive, 432 monthly forecasts per target (confirmed from `results/fredmd/phase1_h1_summary.csv`, `n_obs = 432` for all 41 targets) — use this exact window when this appears in `hrf.tex`, not "Jan 1990."
- RF: `ranger`, `num.trees = 500`, `mtry = round(d/3)`, all other hyperparameters default.
- HRF: κ = 2 fixed. μ̂/Σ̂ via EWMA (λ = 0.15) + linear shrinkage to a constant-variance-covariance target, bandwidth H = 6 (companion paper Appendix D formulas).
- Feature set per rolling window: 4 lags of every complete-history panel column + first 4 PCs of the panel restricted to that window (recomputed per window, not cached) + 4 AR lags of the target's own building block.
- Target treatment (uniform recipe, branch only on `tcode`): `tcode ∈ {5,6}` → building block is the plain one-period percent change `x_t/x_{t-1} - 1` (matching the companion paper's eq. 4.3 exactly — NOT `Δlog(x_t)`, which would make the cumulative-product aggregation below only a first-order approximation instead of an exact inverse), aggregated via cumulative product `x_t · Π(1+block)` (companion paper eq. 4.6, an exact identity for this building block); `tcode = 2` → building block `Δx_t`, aggregate via cumulative sum; `tcode = 1` → no building block, direct multi-lead forecast of the raw value. RMSE/MAE always evaluated on the raw (untransformed) level.
- No test framework is present in this repo (verified: zero files matching `*test*` or referencing `testthat`). This plan introduces a lightweight `tests/timeseries/` convention: plain `Rscript`-executable files using `stopifnot()`, no new dependency.
- Full spec: `docs/superpowers/specs/2026-07-22-fredmd-financial-application-design.md`.

---

### Task 1: FRED-MD tcode transforms and panel loader

**Files:**
- Create: `src/timeseries/get_fredmd_panel.R`
- Test: `tests/timeseries/test_get_fredmd_panel.R`

**Interfaces:**
- Produces: `transform_series(x, tcode)` — numeric vector in, numeric vector out, same length, leading elements `NA` per transform order.
- Produces: `interpolate_na(x)` — numeric vector in, numeric vector out, same length. Fills *internal* `NA` gaps by linear interpolation (`approx()`); leading/trailing `NA` (before a series starts or after it ends) are left as `NA` since there is nothing to interpolate between. Real FRED-MD vintages routinely have isolated single-month gaps near the most recent edge (e.g. this vintage is missing `CPIAUCSL` for 2025-10-01) — this handles that case for every column, not just CPIAUCSL.
- Produces: `load_fredmd_panel(path, start_date = as.Date("1960-01-01"))` — returns a `list(dates, raw, transformed_full, panel, tcodes)`:
  - `dates`: `Date` vector, length T, all rows with `date >= start_date`.
  - `raw`: data.frame, T x V, raw (untransformed) levels for those dates, one column per FRED-MD series (column names = FRED mnemonics), with `interpolate_na()` applied to every column.
  - `transformed_full`: data.frame, T x V, `transform_series` applied to every column using its own tcode, computed on the FULL available history (before restricting to `start_date`, so differencing transforms have the prior row(s) they need), then restricted to the same dates as `raw`.
  - `panel`: data.frame, T x V', the subset of columns of `transformed_full` that have zero `NA` over the full `T` rows (the predictor panel). With the transform-before-restrict ordering above, this should be ~110+ of ~126 columns (matches the companion paper's "111 of 127"); if it comes out much smaller (e.g. ~20), transforms are almost certainly being computed after restricting to `start_date` instead of before.
  - `tcodes`: named numeric vector, tcode per FRED mnemonic (names = column names of `raw`).

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_get_fredmd_panel.R`:

```r
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_fredmd_panel.R`
Expected: FAIL — `Error in source("src/timeseries/get_fredmd_panel.R") : cannot open file` (file doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/get_fredmd_panel.R`:

```r
transform_series <- function(x, tcode) {
  n <- length(x)
  y <- rep(NA_real_, n)
  if (tcode == 1) {
    y <- x
  } else if (tcode == 2) {
    y[2:n] <- x[2:n] - x[1:(n - 1)]
  } else if (tcode == 3) {
    y[3:n] <- x[3:n] - 2 * x[2:(n - 1)] + x[1:(n - 2)]
  } else if (tcode == 4) {
    y <- log(x)
  } else if (tcode == 5) {
    lx <- log(x)
    y[2:n] <- lx[2:n] - lx[1:(n - 1)]
  } else if (tcode == 6) {
    lx <- log(x)
    y[3:n] <- lx[3:n] - 2 * lx[2:(n - 1)] + lx[1:(n - 2)]
  } else if (tcode == 7) {
    y1 <- rep(NA_real_, n)
    y1[2:n] <- (x[2:n] - x[1:(n - 1)]) / x[1:(n - 1)]
    y[3:n] <- y1[3:n] - y1[2:(n - 1)]
  } else {
    stop("Unknown tcode: ", tcode)
  }
  y
}

interpolate_na <- function(x) {
  idx <- seq_along(x)
  ok <- !is.na(x)
  if (sum(ok) < 2) {
    return(x) # nothing usable to interpolate from (all-NA or single valid point)
  }
  x[!ok] <- approx(idx[ok], x[ok], xout = idx[!ok])$y
  x
}

load_fredmd_panel <- function(path, start_date = as.Date("1960-01-01")) {
  raw_csv <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  tcodes <- as.numeric(raw_csv[1, -1])
  names(tcodes) <- colnames(raw_csv)[-1]

  body <- raw_csv[-1, ]
  dates_full <- as.Date(body[[1]], format = "%m/%d/%Y")
  raw_full <- body[, -1, drop = FALSE]
  raw_full[] <- lapply(raw_full, as.numeric)
  raw_full[] <- lapply(raw_full, interpolate_na)

  ## Transform on the FULL available history (back to whatever the file's
  ## earliest date is) BEFORE restricting to start_date. Restricting first
  ## would starve differencing transforms (tcode 2/3/5/6/7) of the prior
  ## row(s) they need, putting an artificial leading NA at the very first
  ## row of the window for every one of those columns and silently
  ## excluding nearly all of them from `panel` below.
  transformed_full_all <- as.data.frame(mapply(
    function(col, tcode) transform_series(col, tcode),
    raw_full, tcodes[colnames(raw_full)],
    SIMPLIFY = FALSE
  ))
  colnames(transformed_full_all) <- colnames(raw_full)

  keep <- dates_full >= start_date
  dates <- dates_full[keep]
  raw <- raw_full[keep, , drop = FALSE]
  transformed_full <- transformed_full_all[keep, , drop = FALSE]
  rownames(transformed_full) <- NULL
  rownames(raw) <- NULL

  complete_cols <- colnames(transformed_full)[colSums(is.na(transformed_full)) == 0]
  panel <- transformed_full[, complete_cols, drop = FALSE]

  list(
    dates = dates,
    raw = raw,
    transformed_full = transformed_full,
    panel = panel,
    tcodes = tcodes
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_fredmd_panel.R`
Expected: prints `transform_series: OK`, `load_fredmd_panel: OK`, `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/timeseries/get_fredmd_panel.R tests/timeseries/test_get_fredmd_panel.R
git commit -m "Add FRED-MD tcode transforms and panel loader"
```

---

### Task 2: Target building-block and aggregation helpers

**Files:**
- Create: `src/timeseries/target_building_block.R`
- Test: `tests/timeseries/test_target_building_block.R`

**Interfaces:**
- Consumes: `transform_series(x, tcode)` from Task 1 (this file must `source("src/timeseries/get_fredmd_panel.R")` itself at the top — do not rely on a caller having sourced it first, so `target_building_block.R` works when sourced standalone).
- Produces: `pct_change(x)` — numeric vector, same length as `x`, `y[1] = NA`, `y[t] = x[t]/x[t-1] - 1` for `t > 1`. This is deliberately distinct from `transform_series(x, 5)` (which computes `Δlog(x)`, a different quantity) — see Global Constraints for why.
- Produces: `get_target_building_block(x, tcode)` — numeric vector, same length as `x`; the one-period-ahead "building block" series used as the direct-forecasting label (see Global Constraints).
- Produces: `aggregate_building_block(x_t, block_forecasts, tcode)` — `x_t`: scalar raw level at the origin date. `block_forecasts`: numeric vector of length `h`, the building-block forecasts for leads `1..h` in order (for `tcode = 1` these are direct raw-level forecasts, not building-block values). Returns a numeric vector of length `h`: raw-level forecasts for leads `1..h`.

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_target_building_block.R`:

```r
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_target_building_block.R`
Expected: FAIL — `cannot open file 'src/timeseries/target_building_block.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/target_building_block.R`:

```r
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_target_building_block.R`
Expected: `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/timeseries/target_building_block.R tests/timeseries/test_target_building_block.R
git commit -m "Add target building-block and raw-level aggregation helpers"
```

---

### Task 3: EWMA + linear shrinkage mean/covariance estimator

**Files:**
- Create: `src/cov_estimators/get_cov_ewma_shrink.R`
- Test: `tests/timeseries/test_get_cov_ewma_shrink.R`

**Interfaces:**
- Produces: `get_cov_ewma_shrink(R, lambda = 0.15, H = 6)` — `R`: numeric matrix, T x p, in-sample tree residuals in **chronological order (row 1 = oldest, row T = most recent)**. Returns `list(mu, sigma)`: `mu` is a length-p numeric vector, `sigma` is a p x p numeric matrix (symmetric, PSD in practice).

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_get_cov_ewma_shrink.R`:

```r
source("src/cov_estimators/get_cov_ewma_shrink.R")

set.seed(42)
Tn <- 60
p <- 5
R <- matrix(rnorm(Tn * p, mean = 0, sd = 1), Tn, p)

result <- get_cov_ewma_shrink(R, lambda = 0.15, H = 6)

stopifnot(is.numeric(result$mu))
stopifnot(length(result$mu) == p)
stopifnot(is.matrix(result$sigma))
stopifnot(all(dim(result$sigma) == c(p, p)))

## symmetry
stopifnot(isTRUE(all.equal(result$sigma, t(result$sigma))))

## diagonal must be strictly positive (variances)
stopifnot(all(diag(result$sigma) > 0))

## PSD: all eigenvalues >= -1e-8 (allow tiny numerical slack)
eig <- eigen(result$sigma, symmetric = TRUE, only.values = TRUE)$values
stopifnot(all(eig > -1e-8))

## shrinkage intensity must strictly shrink toward the constant-variance-covariance
## target: since R is i.i.d. noise here, sigma should differ from the raw (unshrunk)
## weighted sample covariance -- i.e. shrinkage must have actually moved the estimate.
w <- lambda_weights <- 0.15 * (1 - 0.15)^((Tn - 1):0)
xbar <- colMeans(R)
Yc <- sweep(R, 2, xbar, "-")
sigma_hat_unshrunk <- matrix(0, p, p)
for (t in 1:Tn) sigma_hat_unshrunk <- sigma_hat_unshrunk + w[t] * outer(Yc[t, ], Yc[t, ])
stopifnot(!isTRUE(all.equal(result$sigma, sigma_hat_unshrunk)))

## short-window edge case: Tn <= H must not crash (regression guard for a bug
## found in review, where H > Tn - 1 made an internal loop bound count DOWN
## instead of being empty, causing an out-of-bounds access)
R_short <- matrix(rnorm(4 * p), 4, p)
result_short <- get_cov_ewma_shrink(R_short, lambda = 0.15, H = 6)
stopifnot(is.numeric(result_short$mu))
stopifnot(all(is.finite(result_short$sigma)))

cat("ALL PASS\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_cov_ewma_shrink.R`
Expected: FAIL — `cannot open file 'src/cov_estimators/get_cov_ewma_shrink.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/cov_estimators/get_cov_ewma_shrink.R`:

```r
get_cov_ewma_shrink <- function(R, lambda = 0.15, H = 6) {
  Tn <- nrow(R)
  p <- ncol(R)
  # Guard: the h=1..H correction terms need Tn > h observations (D_list[[t-h]]
  # for t up to Tn requires t-h >= 1). With the project's rolling windows
  # (Tn typically in the 340s-350s) this never binds, but without the guard
  # a short window (Tn <= H) makes `(h+1):Tn` count DOWN in R (e.g. 7:5 is
  # c(7,6,5), not empty), causing an out-of-bounds D_list access instead of
  # a clean error or a reduced correction.
  H <- max(0, min(H, Tn - 1))
  w <- lambda * (1 - lambda)^((Tn - 1):0)

  ## --- EWMA mean (D.1) ---
  mu_hat <- as.numeric(colSums(R * w))

  ## --- EWMA covariance (D.2), centered on the SIMPLE mean, per the paper ---
  xbar <- colMeans(R)
  Y <- sweep(R, 2, xbar, "-")
  Sigma_hat <- matrix(0, p, p)
  for (t in seq_len(Tn)) Sigma_hat <- Sigma_hat + w[t] * outer(Y[t, ], Y[t, ])

  ## --- shrinkage target for Sigma: constant-variance-covariance matrix F (D.4) ---
  f1 <- mean(diag(Sigma_hat))
  f2 <- mean(Sigma_hat[upper.tri(Sigma_hat)])
  F_target <- matrix(f2, p, p)
  diag(F_target) <- f1

  ## --- data-dependent shrinkage intensity for Sigma (D.6-D.8), vectorized ---
  D_list <- vector("list", Tn)
  for (t in seq_len(Tn)) D_list[[t]] <- outer(Y[t, ], Y[t, ]) - Sigma_hat

  psi_sum <- function(h) {
    if (h == 0) {
      s <- 0
      for (t in seq_len(Tn)) s <- s + sum(D_list[[t]] * D_list[[t]])
    } else {
      s <- 0
      for (t in (h + 1):Tn) s <- s + sum(D_list[[t]] * D_list[[t - h]])
    }
    s / Tn
  }

  nu_sigma <- (lambda^2 / (1 - (1 - lambda)^2)) *
    (psi_sum(0) + 2 * sum(sapply(seq_len(H), function(h) (1 - lambda)^h * psi_sum(h))))
  gamma_sigma <- sum((F_target - Sigma_hat)^2)
  alpha_sigma <- nu_sigma / (nu_sigma + gamma_sigma)
  Sigma_shrunk <- alpha_sigma * F_target + (1 - alpha_sigma) * Sigma_hat

  ## --- shrinkage target for mu: constant-mean vector (D.5) ---
  mu_star <- mean(mu_hat)

  ## --- data-dependent shrinkage intensity for mu (D.9), cheap per-column acf ---
  nu_i <- sapply(seq_len(p), function(i) {
    psi_i <- as.numeric(acf(R[, i], lag.max = H, type = "covariance", demean = TRUE, plot = FALSE)$acf)
    (lambda^2 / (1 - (1 - lambda)^2)) * (psi_i[1] + 2 * sum((1 - lambda)^(1:H) * psi_i[2:(H + 1)]))
  })
  nu_mu <- sum(nu_i)
  gamma_mu <- sum((mu_star - mu_hat)^2)
  alpha_mu <- nu_mu / (nu_mu + gamma_mu)
  mu_shrunk <- alpha_mu * mu_star + (1 - alpha_mu) * mu_hat

  list(mu = mu_shrunk, sigma = Sigma_shrunk)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_cov_ewma_shrink.R`
Expected: `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/cov_estimators/get_cov_ewma_shrink.R tests/timeseries/test_get_cov_ewma_shrink.R
git commit -m "Add EWMA + linear shrinkage mean/covariance estimator (paper Appendix D)"
```

---

### Task 4: Feature engineering (lags + rolling-window PCs)

**Files:**
- Create: `src/timeseries/build_features.R`
- Test: `tests/timeseries/test_build_features.R`

**Interfaces:**
- Produces: `build_features(panel, target_block, window_idx, n_lags = 4, n_pcs = 4)` —
  `panel`: data.frame, T x V (from Task 1's `load_fredmd_panel()$panel`). `target_block`: numeric vector, length T, aligned row-for-row with `panel` (the target's building-block series over the SAME date range as `panel`, from Task 2's `get_target_building_block()`). `window_idx`: integer vector, length W, contiguous increasing row indices into `panel`/`target_block` defining the current rolling window (`window_idx[1] - n_lags >= 1` required). Returns a data.frame with W rows (row k corresponds to `panel` row `window_idx[k]`) and `n_lags * ncol(panel) + n_pcs + n_lags` columns.

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_build_features.R`:

```r
source("src/timeseries/build_features.R")

set.seed(1)
Tn <- 50
V <- 6
panel <- as.data.frame(matrix(rnorm(Tn * V), Tn, V))
colnames(panel) <- paste0("V", 1:V)
target_block <- rnorm(Tn)

n_lags <- 4
n_pcs <- 4
window_idx <- 20:40 # length 21, window_idx[1]-n_lags = 16 >= 1, OK

feat <- build_features(panel, target_block, window_idx, n_lags = n_lags, n_pcs = n_pcs)

stopifnot(nrow(feat) == length(window_idx))
stopifnot(ncol(feat) == n_lags * V + n_pcs + n_lags)
stopifnot(!any(is.na(feat)))

## spot-check one lag column by hand: V1_lag1 at window row k should equal
## panel$V1 at (window_idx[k] - 1)
stopifnot(all.equal(feat[["V1_lag1"]], panel$V1[window_idx - 1]))
stopifnot(all.equal(feat[["V3_lag4"]], panel$V3[window_idx - 4]))

## spot-check target AR lags
stopifnot(all.equal(feat[["target_lag1"]], target_block[window_idx - 1]))
stopifnot(all.equal(feat[["target_lag4"]], target_block[window_idx - 4]))

## PCs present and finite
stopifnot(all(c("PC1", "PC2", "PC3", "PC4") %in% colnames(feat)))
stopifnot(all(is.finite(feat$PC1)))

cat("ALL PASS\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_build_features.R`
Expected: FAIL — `cannot open file 'src/timeseries/build_features.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/build_features.R`:

```r
build_features <- function(panel, target_block, window_idx, n_lags = 4, n_pcs = 4) {
  if (window_idx[1] - n_lags < 1) {
    stop("Not enough history before the window to build ", n_lags, " lags.")
  }

  panel_mat <- as.matrix(panel)
  V <- ncol(panel_mat)

  lag_blocks <- lapply(seq_len(n_lags), function(L) panel_mat[window_idx - L, , drop = FALSE])
  panel_lags <- do.call(cbind, lag_blocks)
  colnames(panel_lags) <- paste0(rep(colnames(panel_mat), n_lags), "_lag", rep(seq_len(n_lags), each = V))

  window_panel <- panel_mat[window_idx, , drop = FALSE]
  pcs <- prcomp(window_panel, center = TRUE, scale. = TRUE)$x[, seq_len(n_pcs), drop = FALSE]
  colnames(pcs) <- paste0("PC", seq_len(n_pcs))

  target_lag_blocks <- lapply(seq_len(n_lags), function(L) target_block[window_idx - L])
  target_lags <- do.call(cbind, target_lag_blocks)
  colnames(target_lags) <- paste0("target_lag", seq_len(n_lags))

  as.data.frame(cbind(panel_lags, pcs, target_lags))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_build_features.R`
Expected: `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/timeseries/build_features.R tests/timeseries/test_build_features.R
git commit -m "Add lag + rolling-window-PC feature engineering"
```

---

### Task 5: Per-lead RF and HRF forecast

**Files:**
- Create: `src/timeseries/get_forecast_lead.R`
- Test: `tests/timeseries/test_get_forecast_lead.R`

**Interfaces:**
- Consumes: `get_cov_ewma_shrink(R, lambda, H)` from Task 3; `get_weights(mean_vector, cov_matrix, kappa)` from the existing `src/utils/get_weights.R` (unchanged, `sum(w)==1`, `sum(abs(w))<=kappa`, `(w'mu)^2 + w'Sigma w` objective).
- Produces: `get_forecast_lead(features_train, label_train, features_test, num_trees = 500, kappa = 2, ewma_lambda = 0.15, ewma_H = 6)` — `features_train`: data.frame, n x d. `label_train`: numeric vector, length n. `features_test`: data.frame, 1 x d (single row). Returns `list(rf = <scalar>, hrf = <scalar>)`, both building-block-scale (or raw-scale for tcode-1 targets) forecasts for this one lead.

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_get_forecast_lead.R`:

```r
source("src/utils/get_weights.R")
source("src/cov_estimators/get_cov_ewma_shrink.R")
source("src/timeseries/get_forecast_lead.R")

set.seed(7)
n <- 100
d <- 15
X <- as.data.frame(matrix(rnorm(n * d), n, d))
colnames(X) <- paste0("f", 1:d)
y <- X$f1 * 0.5 + rnorm(n, sd = 0.1)
X_test <- X[n, , drop = FALSE]

result <- get_forecast_lead(X[1:(n - 1), , drop = FALSE], y[1:(n - 1)], X_test,
  num_trees = 50, kappa = 2, ewma_lambda = 0.15, ewma_H = 6
)

stopifnot(is.list(result))
stopifnot(all(c("rf", "hrf") %in% names(result)))
stopifnot(is.numeric(result$rf) && length(result$rf) == 1)
stopifnot(is.numeric(result$hrf) && length(result$hrf) == 1)
stopifnot(is.finite(result$rf) && is.finite(result$hrf))

## both forecasts should be in a plausible range given y's scale (not exploding)
stopifnot(abs(result$rf) < 5)
stopifnot(abs(result$hrf) < 5)

cat("ALL PASS\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_forecast_lead.R`
Expected: FAIL — `cannot open file 'src/timeseries/get_forecast_lead.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/get_forecast_lead.R`:

```r
library(ranger)

get_forecast_lead <- function(features_train, label_train, features_test,
                               num_trees = 500, kappa = 2,
                               ewma_lambda = 0.15, ewma_H = 6) {
  mtry <- max(1, round(ncol(features_train) / 3))

  rf <- ranger(x = features_train, y = label_train, num.trees = num_trees, mtry = mtry, keep.inbag = TRUE)

  pred_train_all <- predict(rf, features_train, predict.all = TRUE)$predictions
  pred_test_all <- predict(rf, features_test, predict.all = TRUE)$predictions

  R <- pred_train_all - label_train # in-sample tree residual matrix, chronological order

  mu_sigma <- get_cov_ewma_shrink(R, lambda = ewma_lambda, H = ewma_H)
  w <- get_weights(mu_sigma$mu, mu_sigma$sigma, kappa)

  pred_test_vec <- as.numeric(pred_test_all)
  list(
    rf = mean(pred_test_vec),
    hrf = as.numeric(pred_test_vec %*% w)
  )
}
```

Note: the tree-level forecast error is `e_j := y - M_j(x)` per the paper (`hrf.tex` notation), i.e. `label - prediction`; here `R := pred_train_all - label_train` is its negative (`prediction - label`). This sign flip does not change `get_weights`' solution, since the objective `(w'mu)^2 + w'Sigma w` is invariant to negating the entire residual vector (both `mu` and `sigma` flip/stay consistently), but it must stay consistent with how `hrf.tex`/the companion paper define `R` for any later cross-reference. This is called out explicitly here rather than silently — if this ever needs to match the paper's sign convention exactly, negate `R` (`label_train - pred_train_all`) instead; the weights are numerically identical either way because `mu -> -mu` and `Sigma` unchanged leaves `(w'mu)^2` unchanged and `w'Sigma w` unchanged, so `get_weights` returns the same `w`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_forecast_lead.R`
Expected: `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/timeseries/get_forecast_lead.R tests/timeseries/test_get_forecast_lead.R
git commit -m "Add per-lead RF/HRF forecast function"
```

---

### Task 6: Vintage/horizon forecast driver

**Files:**
- Create: `src/timeseries/get_forecast_vintage_ts.R`
- Test: `tests/timeseries/test_get_forecast_vintage_ts.R`

**Interfaces:**
- Consumes: `build_features()` (Task 4), `get_forecast_lead()` (Task 5), `aggregate_building_block()` (Task 2).
- Produces: `get_forecast_vintage_ts(panel, target_block, target_raw, origin_idx, horizons, tcode, window_length = 360, n_lags = 4, n_pcs = 4, num_trees = 500, kappa = 2, ewma_lambda = 0.15, ewma_H = 6, winsorize_prob = 0.01)` —
  `panel`: data.frame, T x V (Task 1). `target_block`: numeric vector, length T, aligned with `panel` rows (Task 2's building block for this target). `target_raw`: numeric vector, length T, the target's raw level, aligned with `panel` rows. `origin_idx`: integer, the row index of the forecast origin (must satisfy `origin_idx - window_length + 1 - n_lags >= 1`). `horizons`: integer vector, e.g. `1` or `1:12`, the horizons to forecast (all `<= max(horizons)` leads get fit once and reused). `tcode`: the target's own tcode (1, 2, 5, or 6).
  Returns a data.frame with one row per horizon in `horizons`, columns `horizon`, `rf_forecast`, `hrf_forecast`, `actual` (all in raw units) — `actual` is `NA` if `origin_idx + horizon` exceeds `length(target_raw)`.

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_get_forecast_vintage_ts.R`:

```r
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_forecast_vintage_ts.R`
Expected: FAIL — `cannot open file 'src/timeseries/get_forecast_vintage_ts.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/get_forecast_vintage_ts.R`:

```r
get_forecast_vintage_ts <- function(panel, target_block, target_raw, origin_idx, horizons, tcode,
                                     window_length = 360, n_lags = 4, n_pcs = 4,
                                     num_trees = 500, kappa = 2,
                                     ewma_lambda = 0.15, ewma_H = 6,
                                     winsorize_prob = 0.01) {
  window_idx <- (origin_idx - window_length + 1):origin_idx
  features <- build_features(panel, target_block, window_idx, n_lags = n_lags, n_pcs = n_pcs)
  W <- nrow(features)

  max_h <- max(horizons)
  block_rf <- numeric(max_h)
  block_hrf <- numeric(max_h)

  for (j in seq_len(max_h)) {
    n_train <- W - j
    features_train <- features[seq_len(n_train), , drop = FALSE]
    label_train <- target_block[window_idx[(1 + j):W]]

    if (winsorize_prob > 0) {
      qs <- quantile(label_train, probs = c(winsorize_prob, 1 - winsorize_prob), na.rm = TRUE)
      label_train <- pmin(pmax(label_train, qs[1]), qs[2])
    }

    features_test <- features[W, , drop = FALSE]

    lead_result <- get_forecast_lead(features_train, label_train, features_test,
      num_trees = num_trees, kappa = kappa, ewma_lambda = ewma_lambda, ewma_H = ewma_H
    )
    block_rf[j] <- lead_result$rf
    block_hrf[j] <- lead_result$hrf
  }

  x_t <- target_raw[origin_idx]
  raw_rf <- aggregate_building_block(x_t, block_rf, tcode)
  raw_hrf <- aggregate_building_block(x_t, block_hrf, tcode)

  Tn <- length(target_raw)
  data.frame(
    horizon = horizons,
    rf_forecast = raw_rf[horizons],
    hrf_forecast = raw_hrf[horizons],
    actual = ifelse(origin_idx + horizons <= Tn, target_raw[pmin(origin_idx + horizons, Tn)], NA_real_)
  )
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_get_forecast_vintage_ts.R`
Expected: `tcode 1 OK`, `tcode 2 OK`, `tcode 6 OK`, `ALL PASS`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/timeseries/get_forecast_vintage_ts.R tests/timeseries/test_get_forecast_vintage_ts.R
git commit -m "Add vintage/horizon forecast driver with tcode-based aggregation"
```

---

### Task 7: Target registry and scan runner

**Files:**
- Create: `src/timeseries/fredmd_targets.R`
- Create: `src/timeseries/run_fredmd_scan.R`
- Create: `main_fredmd_scan.R`
- Test: `tests/timeseries/test_fredmd_targets.R`

**Interfaces:**
- Produces (`fredmd_targets.R`): `fredmd_targets` — a data.frame with columns `id` (FRED mnemonic, character), `group` (one of `"Prices"`, `"Rate"`, `"Spread"`, `"FX"`), `tcode` (integer, one of 1/2/5/6). 41 rows.
- Produces (`run_fredmd_scan.R`): `run_fredmd_scan(fredmd_data, targets, horizons, oos_start_date, mc_cores = 30, ...)` — `fredmd_data`: the `list` returned by `load_fredmd_panel()`. `targets`: a data.frame like `fredmd_targets` (or a subset of its rows). `horizons`: integer vector. `oos_start_date`: `Date`. `...`: forwarded to `get_forecast_vintage_ts()` (e.g. `window_length`, `num_trees`). Returns a data.frame with columns `id`, `horizon`, `origin_date`, `rf_forecast`, `hrf_forecast`, `actual` — one row per (target, valid OOS origin date, horizon).

- [ ] **Step 1: Write the failing test**

Create `tests/timeseries/test_fredmd_targets.R`:

```r
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_fredmd_targets.R`
Expected: FAIL — `cannot open file 'src/timeseries/fredmd_targets.R'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/timeseries/fredmd_targets.R`:

```r
fredmd_targets <- data.frame(
  id = c(
    # Prices (tcode 6) -- 20 series
    "WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "OILPRICEx", "PPICMM",
    "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAD",
    "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "PCEPI",
    "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA",
    # Rate levels (tcode 2) -- 9 series
    "FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA",
    # Credit/term spreads (tcode 1) -- 8 series
    "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM",
    # FX rates (tcode 5) -- 4 series
    "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx"
  ),
  group = c(
    rep("Prices", 20),
    rep("Rate", 9),
    rep("Spread", 8),
    rep("FX", 4)
  ),
  tcode = c(
    rep(6, 20),
    rep(2, 9),
    rep(1, 8),
    rep(5, 4)
  ),
  stringsAsFactors = FALSE
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/hedged-random-forest && Rscript tests/timeseries/test_fredmd_targets.R`
Expected: `ALL PASS`, exit code 0.

- [ ] **Step 5: Write `run_fredmd_scan.R`**

Create `src/timeseries/run_fredmd_scan.R`:

```r
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
```

- [ ] **Step 6: Write `main_fredmd_scan.R`**

Create `main_fredmd_scan.R`:

```r
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
```

- [ ] **Step 7: Commit**

```bash
git add src/timeseries/fredmd_targets.R src/timeseries/run_fredmd_scan.R main_fredmd_scan.R tests/timeseries/test_fredmd_targets.R
git commit -m "Add target registry, scan runner, and Phase-1 entry script"
```

---

### Task 8: Run the Phase-1 (h=1) scan and inspect results

**Files:**
- None created; this task executes `main_fredmd_scan.R` and reviews its output.

- [ ] **Step 1: Smoke-test on a tiny slice before the full run**

Run a quick interactive check that the full pipeline works end-to-end on real data before committing to the ~45-60 minute full scan:

```bash
cd /home/ubuntu/hedged-random-forest
Rscript -e '
source("src/utils/get_weights.R")
source("src/timeseries/get_fredmd_panel.R")
source("src/timeseries/target_building_block.R")
source("src/cov_estimators/get_cov_ewma_shrink.R")
source("src/timeseries/build_features.R")
source("src/timeseries/get_forecast_lead.R")
source("src/timeseries/get_forecast_vintage_ts.R")
source("src/timeseries/fredmd_targets.R")

fredmd_data <- load_fredmd_panel("data/hrf-ts/2026-06-MD.csv", start_date = as.Date("1960-01-01"))
origin_idx <- which(fredmd_data$dates == as.Date("1990-01-01"))
target_row <- fredmd_targets[fredmd_targets$id == "FEDFUNDS", ]
target_raw <- fredmd_data$raw[["FEDFUNDS"]]
target_block <- get_target_building_block(target_raw, target_row$tcode)

t0 <- Sys.time()
result <- get_forecast_vintage_ts(
  fredmd_data$panel, target_block, target_raw, origin_idx, horizons = 1, tcode = target_row$tcode
)
cat("elapsed:", as.numeric(Sys.time() - t0, units = "secs"), "sec\n")
print(result)
'
```

Expected: completes in a few seconds, prints one row with `horizon=1`, finite `rf_forecast`/`hrf_forecast` numerically close to the actual FEDFUNDS level around Feb 1990 (a few percentage points), and no errors.

- [ ] **Step 2: Run the full Phase-1 scan**

```bash
cd /home/ubuntu/hedged-random-forest
time Rscript main_fredmd_scan.R 2>&1 | tee results/fredmd/phase1_run.log
```

Expected: runs for roughly 45-90 minutes (revised estimate from the ~2.6s/call EWMA-shrinkage benchmark and ~437 OOS months x 41 targets, parallelized across 30 cores), ends by printing the `phase1_summary` data.frame (41 rows: `id`, `rmse_ratio`, `mae_ratio`, `n_obs`, `group`, `tcode`) with no errors, and writes `results/fredmd/phase1_h1_scan.rds` and `results/fredmd/phase1_h1_summary.csv`.

- [ ] **Step 3: Inspect and sanity-check the results**

```bash
Rscript -e '
s <- read.csv("results/fredmd/phase1_h1_summary.csv")
cat("targets with rmse_ratio < 1:", sum(s$rmse_ratio < 1), "of", nrow(s), "\n")
print(s[order(s$rmse_ratio), c("id", "group", "rmse_ratio", "mae_ratio", "n_obs")])
'
```

Expected: `n_obs` close to 437 for every target (confirms the OOS window is being used fully); a mix of ratios, most or many below 1 per the companion paper's precedent — this table is the actual deliverable that determines which targets go into `hrf.tex` (Section 6 of the design spec), not something to force toward a particular outcome.

- [ ] **Step 4: Commit results**

```bash
git add results/fredmd/phase1_h1_summary.csv
git commit -m "Add Phase-1 (h=1) FRED-MD scan results"
```

(`results/fredmd/phase1_h1_scan.rds` and `phase1_run.log` are large/log artifacts — leave them untracked unless the repo's existing convention for `results/` tracks binary `.rds` files too; check `git status` after Task 8 Step 2 and follow whatever the surrounding `results/` directory already does before deciding.)

---

## What this plan does not cover (by design, per the spec's Section 8)

- Phase 2 (full `h=1..12` grid on the curated winning subset) — depends on Phase 1 results.
- Selecting the headline target(s) and drafting the new `hrf.tex` subsection — depends on Phase 1/2 results.
- Any κ-sweep, alternative λ, or other robustness checks — the spec fixes these to the companion paper's defaults for this first pass.
