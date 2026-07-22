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
