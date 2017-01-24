# This is analysis based on Deep Learning Network build with TensorFlow framework.

source('./src/config.R')
source('./src/dnn.R')
source('./src/users_likes_data_set.R')

# make sure that library installed
library(tensorflow)

# Basic model parameters as external flags.
flags <- tf$app$flags
flags$DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags$DEFINE_integer('max_steps', 5000L, 'Number of steps to run trainer.')
flags$DEFINE_integer('hidden1', 256L, 'Number of units in hidden layer 1.')
flags$DEFINE_integer('hidden2', 64L, 'Number of units in hidden layer 2.')
flags$DEFINE_integer('batch_size', 100L, 'Batch size. Must divide evenly into the dataset sizes.')
flags$DEFINE_string('train_dir', sprintf("%s/train_data", out_intermediates_dir), 'Directory to put the training data.')
FLAGS <- parse_flags()

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_prdata_file))

# 
# Load data
input_data = ULDataSet$new()

# set random number generator's seed - so results will be stable from run to run
set.seed(44)

# Returns next random batch from data set
#
# Args:
#   batch_size: The batch size will be baked into both placeholders.
#
# Returns:
#   batch$user_likes: The list of train data inputs
#   batch$user_traits: the list of train ground truth data
next_batch <- function(batch_size) {
  stop("Needed definition of method here")
}


# Generate placeholder variables to represent the input tensors.
#
# These placeholders are used as inputs by the rest of the model building
# code and will be fed from the downloaded data in the .run() loop, below.
#
# Args:
#   batch_size: The batch size will be baked into both placeholders.
#
# Returns:
#   placeholders$user_likes: the Users-Likes placeholder.
#   placeholders$user_traits: the user traits placeholder.
#
placeholder_inputs <- function(batch_size, features_dimensions) {
  
  # Note that the shapes of the placeholders match the shapes of the full
  # User-Likes and user traits tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  user_likes <- tf$placeholder(tf$float32, shape(batch_size, features_dimensions))
  user_traits <- tf$placeholder(tf$int32, shape(batch_size, OUTPUTS_DIMENSION))
  
  # return both placeholders
  list(user_likes = user_likes, user_traits = user_traits)
}

# Fills the feed_dict for training the given step.
#
# A feed_dict takes the form of:
#   feed_dict = dict(
#     <placeholder = <tensor of values to be passed for placeholder>,
#     ....
#   )
#
# Args:
#   data_set: The set of user likes and user traits, from input_data.read_data_sets()
#   user_likes_pl: the Users-Likes placeholder, from placeholder_inputs().
#   user_traits_pl: the user traits placeholder, from placeholder_inputs().
#
# Returns:
#   feed_dict: The feed dictionary mapping from placeholders to values.
#
fill_feed_dict <- function(data_set, user_likes_pl, user_traits_pl) {
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  batch <- data_set$next_batch(FLAGS$batch_size)
  dict(
    user_likes_pl = batch$user_likes,
    user_traits_pl = batch$user_traits
  )
}

# Train users psychometric model

# Get sets of users-likes and users traits for train and test
data_sets <- input_data$read_data_set(ul_file = ul_prdata_file, users_file = users_prdata_file)






