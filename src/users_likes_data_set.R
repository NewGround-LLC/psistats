# The Users-Likes data set processing
#
library(R6)

# The data set factory
ul.read_data_set <- function(ul_file, users_file) {
  load(users_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n","Users", dim(users)[1], dim(users)[2]))
  load(ul_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n", "Users-Likes", dim(M)[1], dim(M)[2]))
  
  # prepare data
  #private$users_clear <- users[,-1]
  users_clear <- users
  users_clear['age'] <- scale(users['age'])
  features_dimension <- dim(M)[2]
  
  folds <- sample(1:n_folds, size = nrow(users), replace = TRUE) # the data samples folds for cross-validation
  test <- folds == 1
  
  # create train/test data sets
  train <- ULDataSet$new(users = users_clear[!test], users_likes = M[!test])
  test <- ULDataSet$new(users = users_clear[test], users_likes = M[test])
  
  # release memory
  rm(M, users)
  
  # create list and return
  list(
    "train" <- train,
    "test" <- test,
    "features_dimension" <- features_dimension
  )
}

# The Users-Likes psychodemographics data set holder class
ULDataSet <- R6Class(
  "ULDataSet",
  public = list(
    # The current active index
    active_index = NA,
    # Initialize
    initialize = function(users, users_likes) {
      private$users = users
      private$users_likes = users_likes
    },
    next_batch = function(batch_size) {
      stop("Method not implemented")
    }
  ),
  private = list(
    users = NA,
    users_likes = NA
  )
) 