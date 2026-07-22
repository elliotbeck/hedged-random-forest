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

load_fredmd_panel <- function(path, start_date = as.Date("1960-01-01")) {
  raw_csv <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  tcodes <- as.numeric(raw_csv[1, -1])
  names(tcodes) <- colnames(raw_csv)[-1]

  body <- raw_csv[-1, ]
  dates <- as.Date(body[[1]], format = "%m/%d/%Y")
  raw <- body[, -1, drop = FALSE]
  raw[] <- lapply(raw, as.numeric)

  keep <- dates >= start_date
  dates <- dates[keep]
  raw <- raw[keep, , drop = FALSE]

  transformed_full <- as.data.frame(mapply(
    function(col, tcode) transform_series(col, tcode),
    raw, tcodes[colnames(raw)],
    SIMPLIFY = FALSE
  ))
  colnames(transformed_full) <- colnames(raw)
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
