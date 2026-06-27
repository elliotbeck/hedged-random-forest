mae <- function(predictions, labels) {
  mean(abs(predictions - labels))
}
