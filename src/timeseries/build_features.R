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
