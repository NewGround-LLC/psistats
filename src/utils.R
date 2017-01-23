# The required libraries
library(ROCR)

# The function to plot scatter plot among with fitted line
plot.fitted <- function(x, y, name, color = "red") {
  # Unlist input
  us <- unlist(y)
  
  # Generate first order linear model
  # lin.mod <- lm(us ~ x)
  # Generate second order linear model
  # lin.mod2 <- lm(us ~ I(x ^ 2) + x)
  # Calculate local regression
  ls <- loess(us ~ x)
  
  # Predict the data to fit
  # pr.lm <- predict(lin.mod)
  # pr.lm2 <- predict(lin.mod2)
  pr.loess <- predict(ls)
  
  # Plot points
  plot(x, us, type = "p", las = 1, xlab = "K", ylab = "r")
  # Plot fit line
  lines(pr.loess~x, col = color, lwd = 2)
  title(main = name)
}

# The function to make linear/logistic regression predictions
linear.fit.predict <- function(response, column, data, testFold) {
  # check if variable is binary (0, 1 - gender in our data samples)
  if(length(unique(na.omit(response[,column]))) == 2) {
    # use logistic regression for binominal classification
    fit <- glm(response[,column]~., data = data, subset = !testFold, family = "binomial")
    return (predict(fit, data[testFold, ], type = "response")) # store predictions in test indices
  } else {
    # use linear regression to directly estimate variable value
    fit<-glm(response[,column]~., data = data, subset = !testFold)
    return (predict(fit, data[testFold, ]))
  }
}

# The function to calculate accuracy
accuracy <- function(groundTruth, Y) {
  if (length(unique(na.omit(groundTruth))) == 2) {
    # The binominal classification results - check the area under 
    # the receiver-operating characteristics curve (AUC ROC)
    f <- which(!is.na(groundTruth))
    temp <- prediction(Y[f], groundTruth[f])
    return (performance(temp,"auc")@y.values)
  } else {
    # The continuous values prediction results
    return (cor(groundTruth, Y, use = "pairwise"))
  }
}

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
  cat("\n\n")
}

trimMatrix <- function(M, minUsersPerLike, minLikesPerUser) {
  repeat {
    i <- sum(dim(M))
    M <- M[rowSums(M) >= minLikesPerUser, colSums(M) >= minUsersPerLike]
    if (sum(dim(M)) == i) break # nothing was removed - finished
  }
  M
}
