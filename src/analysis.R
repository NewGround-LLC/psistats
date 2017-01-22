# This is preliminary analysis to find correlations between users scores on varimax rotated SVD dimensions 
# and users' psychodemographic traits. The matrix with users scores on varimax rotated SVD dimensions
# generated during analysis will be saved and can be used to build corresponding ML models in other
# modules.

source('./src/config.R')

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

#
# Prepare samples for k-fold cross-validation 
folds <- sample(1:10, size = nrow(users), replace = TRUE)
# set appart test data samples
test <- folds == 1

#
# Perform dimensionality reduction with SVD for 50 dimensions
K <- 50
Msvd <- irlba(M[!test,], nv = K)
# Get varimax rotated scores for Likes per SVD dimensions
likesSVDrot <- unclass(varimax(Msvd$v)$loadings)
# Get varimax rotated scores for Users per SVD dimensions
usersSVDrot <- as.data.frame(as.matrix(M %*% likesSVDrot))

print("Dimensionality reduction complete")

# save calculated users scores and cross-validation folds
save(usersSVDrot, file = users_rot_varimax_file)
save(folds, file = users_cv_folds_file)

# find correlations between users scores on varimax rotated SVD dimensions and users psychodemographic traits
users_svd_corr <- cor(usersSVDrot, users[, -1], use = "pairwise.complete.obs")

# plot it as heatmap for analysis
heatmap(users_svd_corr)
