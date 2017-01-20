# This is implementation of linear/logistic regression analysis to build prediction models
# based on FB likes of users vs corresponding psyhometric traits. This models will allow
# to predict psychometric traits of users taking as input their FB likes.

source('./src/config.R')

# Make sure that SVD library installed
library(irlba)

# Check that input data exist
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_prdata_file))
  
# Load data
users <- load(users_prdata_file)
M <- load(ul_prdata_file)