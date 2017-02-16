# The Users-Likes data set processing
#
library(R6)
library(irlba)

source('./src/config.R')

# The function to read preprocessed data set from saved SVD Users-Likes matrix and Users traits list.
#
# Args:
#   ul_file: the file with Users-Likes sparse matrix
#   users_file: the file with Users traits table
# Returns:
#   list comprising of two data sets (train, test) and number of features in data samples (features_dimension)
ul_read_data_set <- function(ul_file, users_file) {
  load(users_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n","Users", dim(users)[1], dim(users)[2]))
  load(ul_file)
  cat(sprintf("%12s : [%d, %d]\n", "Users-Likes", dim(M)[1], dim(M)[2]))
  
  # prepare data
  folds <- sample(1:n_folds, size = nrow(users), replace = TRUE) # the data samples folds for cross-validation
  test <- folds == 1
  
  # create train/test data sets
  train <- ULDataSet$new(users = users[!test,], users_likes = M[!test,])
  test <- ULDataSet$new(users = users[test,], users_likes = M[test,])
  features_dimension <- dim(M)[2]
  
  # create list and return
  list(
    train = train,
    test = test,
    features_dimension = features_dimension
  )
}

# The preprocessing function to reduce features dimension with SVD in Users-Likes matrix and
# save resulting matrix for further use.
#
# Args:
#   ul_file: the file with full features sparse Users-Likes matrix
#   out_file: the file to store preprocessing results
#   svd_k: the number of SVD dimensions [default: 50]
#   varimax_rotate: the flag to indicate whether results of factor analysis (SVD) should be varimax rotated
ul_save_features_reduced_data_set <- function(ul_file, out_file, svd_k = 50, varimax_rotate = TRUE) {
  load(ul_prdata_file)
  cat(sprintf("%12s : [%d, %d]\n", "Users-Likes", dim(M)[1], dim(M)[2]))
  
  # do SVD with varimax rotation
  start_time <- Sys.time()
  Msvd <- irlba(M, nv = svd_k)
  if (varimax_rotate) {
    likesSVDrot <- unclass(varimax(Msvd$v)$loadings) # varimax rotated likes per SVD dimensions
    M <- as.data.frame(as.matrix(M %*% likesSVDrot)) # the varimax rotated users-likes per SVD dimensions
  } else {
    M <- as.data.frame(as.matrix(Msvd$u)) # the users per SVD dimensions
  }
    
  duration <- Sys.time() - start_time
  # save reduced matrix if requested
  save(M, file = out_file)
  
  cat(sprintf('SVD with varimax rotation complete, dimensions: %d, varimax: %s, duration: %s, resulting file: %s\n',
              svd_k, varimax_rotate, duration, out_file))
}

# The Users-Likes psychodemographics data set holder class
ULDataSet <- R6Class(
  "ULDataSet",
  public = list(
    # The current active index
    active_index = NA,
    # The number of data examples
    num_examples = NA,
    # The column names of target data
    labels_names = NA,
    # The batch indices selected during last sampling
    batch_sampling_indices = NA,
    # Initialize
    initialize = function(users, users_likes) {
      private$users = users
      private$users_likes = users_likes
      self$num_examples = dim(users)[1]
      self$labels_names = colnames(users)
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
      self$batch_sampling_indices = sample(1:self$num_examples, size = batch_size, replace = TRUE)
      list(
        users_likes = private$users_likes[self$batch_sampling_indices,],
        users = private$users[self$batch_sampling_indices,]
      )
    }
  ),
  private = list(
    users = NA,
    users_likes = NA
  )
) 