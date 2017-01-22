# This is preliminary analysis to find correlations between users scores on varimax rotated SVD dimensions 
# and users' psychodemographic traits. As results of analysis heatmap will be generated as visual representation
# of mentioned correlation.

source('./src/config.R')
source('./src/utils.R')

# Make sure that SVD library installed
library(irlba)

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

# set random number generator seed - so results will be stable from run to run
set.seed(44)

# the data samples folds for cross-validation
folds <- sample(1:n_folds, size = nrow(users), replace = TRUE)
test <- folds == 1

#
# Perform dimensionality reduction with SVD for 50 dimensions
Msvd <- irlba(M[!test,], nv = K)
# Get varimax rotated scores for Likes per SVD dimensions
likesSVDrot <- unclass(varimax(Msvd$v)$loadings)
# Get varimax rotated scores for Users per SVD dimensions
usersSVDrot <- as.data.frame(as.matrix(M %*% likesSVDrot))

print("Dimensionality reduction complete")

# find correlations between users scores on varimax rotated SVD dimensions and users psychodemographic traits
users_svd_corr <- cor(usersSVDrot, users[, -1], use = "pairwise.complete.obs")
# plot it as heatmap for analysis
heatmap(users_svd_corr)

#
# Do measure prediction performance for variable
vars <- colnames(users[,-1])
ks <- c(2:10, 15, 20, 30, 40, 50) # the numbers of SVD dimensions to test against
for(var in vars) {
  print(sprintf("Measure performance for: [%s]", var))
  rs <- list() # the results to hold data
  for(i in ks) {
    likesSVDrot <- unclass(varimax(Msvd$v[,1:i])$loadings)
    usersSVDrot <- as.data.frame(as.matrix(M %*% likesSVDrot))
    
    pred <- linear.fit.predict(response = users, column = var, data = usersSVDrot, testFold = test)
    rs[[as.character(i)]] <- accuracy(users[,var][test], pred)
  }
  # Plot
  plot.fitted(x = ks, y = rs, name = var)
}







