# This is implementation of linear/logistic regression analysis to build prediction models
# based on FB likes of users vs corresponding psyhometric traits. This models will allow
# to predict psychometric traits of users taking as input their FB likes.

source('./src/config.R')
source('./src/utils.R')

# Make sure that SVD and ROCR libraries installed
library(irlba)
library(ROCR)

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_prdata_file))

# 
# Load data
load(users_prdata_file)
cat("Users:\t\t", dim(users), "\n")
load(ul_prdata_file)
cat("Users-Likes:\t\t", dim(M), "\n")

# set random number generator's seed - so results will be stable from run to run
set.seed(44)

#
# Prepare parameters for regression analysis
rVars <- colnames(users[,-1]) # the variables to predict
folds <- sample(1:n_folds, size = nrow(users), replace = TRUE) # the data samples folds for cross-validation

# Fill predictions list with NA
predictions <- list()
binary_vars <- list()
for(var in rVars) {
  predictions[[var]] <- rep(NA, n = nrow(users))
  if(length(unique(na.omit(users[,var]))) == 2){
    binary_vars[[var]] <- TRUE
  } else {
    binary_vars[[var]] <- FALSE
  }
}

#
# Do cross-validated predictions
for(i in 1:n_folds) {
  print(sprintf("Cross-validated prediction, fold: %d", i))
  test <- folds == i # select test samples
  
  # do SVD rotation
  Msvd <- irlba(M[!test, ], nv = K)
  likesSVDrot <- unclass(varimax(Msvd$v)$loadings) # varimax rotated likes per SVD dimensions
  usersSVDrot <- as.data.frame(as.matrix(M %*% likesSVDrot))
  
  for(var in rVars) {
    # check if variable is binary (0, 1 - gender in our data samples)
    if(binary_vars[[var]]) {
      # use logistic regression for binominal classification
      fit <- glm(users[,var]~., data = usersSVDrot, subset = !test, family = "binomial")
      predictions[[var]][test] <- predict(fit, usersSVDrot[test, ], type = "response") # store predictions in test indices
    } else {
      # use linear regression to directly estimate variable value
      fit<-glm(users[,var]~., data = usersSVDrot, subset = !test)
      predictions[[var]][test] <- predict(fit, usersSVDrot[test, ])
    }
    print(sprintf("Model for variable [%s] complete", var))
  }
  cat("\n***\n")
}

#
# Find accuracies for all predicted variables
print("Prediction accuracies")
accuracies <- list()
for(var in rVars) {
  accuracies[[var]] = accuracy(users[,var], predictions[[var]])
  cat(sprintf("%9s : %.2f%%\n", var, (accuracies[[var]][[1]] * 100.0)))
}



