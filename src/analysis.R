# This is preliminary analysis to find correlations between users scores on varimax rotated SVD dimensions 
# and users' psychodemographic traits. As results of analysis heatmap will be generated as visual representation
# of mentioned correlation.

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

# set random number generator seed - so results will be stable from run to run
set.seed(44)

#
# Perform dimensionality reduction with SVD for 50 dimensions
Msvd <- irlba(M, nv = K)
# Get varimax rotated scores for Likes per SVD dimensions
likesSVDrot <- unclass(varimax(Msvd$v)$loadings)
# Get varimax rotated scores for Users per SVD dimensions
usersSVDrot <- as.data.frame(as.matrix(M %*% likesSVDrot))

print("Dimensionality reduction complete")

# find correlations between users scores on varimax rotated SVD dimensions and users psychodemographic traits
users_svd_corr <- cor(usersSVDrot, users[, -1], use = "pairwise.complete.obs")

# plot it as heatmap for analysis
heatmap(users_svd_corr)
