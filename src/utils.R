# The value formatter for summary report
formatValue <- function(val, digits = 10, nsmall = 0) {
  format(val, width = 12, justify = "right", digits = digits, nsmall= nsmall, big.mark = " ")
}

# prints summary of provided array as vertical table
printSummaryVertical <- function(arr) {
  cat("\tMean\t\t", formatValue(mean(arr), digits = 0), "\n")
  cat("\tMedian\t\t", formatValue(median(arr)), "\n")
  cat("\tMinimum\t\t", formatValue(min(arr)), "\n")
  cat("\tMaximum\t\t", formatValue(max(arr)))
}

# Print the matrix summary report
printULSummary <- function(M) {
  # General statistics
  ulPairs <- sum(attr(M, "x"))
  usersCount <- attr(M, "Dim")[1]
  likesCount <- attr(M, "Dim")[2]
  matrixDensity <- ulPairs / (as.numeric(usersCount) * as.numeric(likesCount)) * 100
  
  cat("# of users\t\t", formatValue(usersCount))
  cat("\n# of unique Likes\t", formatValue(likesCount))
  cat("\n# of User-Like pairs\t", formatValue(ulPairs))
  cat("\nMatrix density\t\t", formatValue(matrixDensity, digits = 0, nsmall = 3), "%")
  
  # Likes per User
  likesPerUser <- rowSums(M)
  cat("\nLikes per User\n")
  printSummaryVertical(likesPerUser)
  
  # Users per Like
  usersPerLike <- colSums(M)
  cat("\nUsers per Like\n")
  printSummaryVertical(usersPerLike)
}

trimMatrix <- function(M, minUsersPerLike, minLikesPerUser) {
  repeat {
    i <- sum(dim(M))
    M <- M[rowSums(M) >= minLikesPerUser, colSums(M) >= minUsersPerLike]
    if (sum(dim(M)) == i) break # nothing was removed - finished
  }
  M
}
