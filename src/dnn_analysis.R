# This is analysis based on Deep Learning Network build with TensorFlow framework.

# make sure that library installed
library(tensorflow)

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_prdata_file))

# 
# Load data
load(users_prdata_file)
cat(sprintf("%12s : [%d, %d]\n","Users", dim(users)[1], dim(users)[2]))
load(ul_prdata_file)
cat(sprintf("%12s : [%d, %d]\n", "Users-Likes", dim(M)[1], dim(M)[2]))

#
# Build computation graph
