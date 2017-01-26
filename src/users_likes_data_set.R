# The Users-Likes data set processing
#
library(R6)
library(irlba)

# The data set factory to load and generate data set from provided files
#
# Args:
#   ul_file: the file with Users-Likes sparse matrix
#   users_file: the file with Users traits table
#   with_svd: the flag to indicate if SVD should be applied to reduce features dimension [default: false]
#   svd_k: the number of SVD dimensions [default: 50]
# Returns:
#   list comprising of two data sets (train, test) and number of features in data samples (features_dimension)
ul.read_data_set <- function(ul_file, users_file, with_svd = FALSE, svk_k = 50) {
  load(users_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n","Users", dim(users)[1], dim(users)[2]))
  load(ul_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n", "Users-Likes", dim(M)[1], dim(M)[2]))
  
  # prepare data
  folds <- sample(1:n_folds, size = nrow(users), replace = TRUE) # the data samples folds for cross-validation
  test <- folds == 1
  
  # Apply SVD if appropriate
  if(with_svd) {
    # do SVD rotation
    Msvd <- irlba(M[!test, ], nv = svk_k)
    likesSVDrot <- unclass(varimax(Msvd$v)$loadings) # varimax rotated likes per SVD dimensions
    M <- as.data.frame(as.matrix(M %*% likesSVDrot)) # the varimax rotated users-likes per SVD dimensions
  }

  # create train/test data sets
  train <- ULDataSet$new(users = users[!test,], users_likes = M[!test,])
  test <- ULDataSet$new(users = users[test,], users_likes = M[test,])
  features_dimension <- dim(M)[2]
  
  # release memory
  rm(M, users)
  
  # create list and return
  list(
    train = train,
    test = test,
    features_dimension = features_dimension
  )
}

# The Users-Likes psychodemographics data set holder class
ULDataSet <- R6Class(
  "ULDataSet",
  public = list(
    # The current active index
    active_index = NA,
    # The number of data examples
    num_examples = NA,
    # Initialize
    initialize = function(users, users_likes) {
      private$users = users
      private$users_likes = users_likes
      self$num_examples = dim(users)[1]
    },
    # Method to return batch of data with specified size
    #
    # Args:
    #   batch_size: the size of data batch to return
    # Returns:
    #   list with data batch including:
    #     user_likes: the sparse matrix of users-likes
    #     users: the list with users' data
    next_batch = function(batch_size) {
      assertthat::assert_that(batch_size < self$num_examples)
      # find next sample and return list with next batch of data
      batch_indx = sample(1:self$num_examples, size = batch_size, replace = TRUE)
      list(
        users_likes = private$users_likes[batch_indx,],
        users = private$users[batch_indx,]
      )
    }
  ),
  private = list(
    users = NA,
    users_likes = NA
  )
) 