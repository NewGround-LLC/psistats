# This is analysis based on fast forward fully connected artificial neural network with dropout regularization 
# build with TensorFlow framework.

source('./src/config.R')
source('./src/users_likes_data_set.R')
source('./src/utils.R')

# make sure that library installed
library(tensorflow)
library(optparse)

# Basic model parameters as external flags.
option_list <- list(
  make_option(c("--learning_rate"), type="double", default=0.001,
              help="Initial learning rate. [default %default]"),
  make_option(c("--max_steps"), type="integer", default=50000L,
              help="Number of steps to run trainer. [default %default]"),
  make_option(c("--layers"), type="character", #default="512",
              help="Specify number of neurons per layer separated by coma (e.g.: 512,256,128)."),
  make_option(c("--batch_size"), type="integer", default=100L,
              help="Batch size. Must divide evenly into the dataset sizes. [default %default]"),
  make_option(c("--train_dir"), type="character", default=sprintf("%s/train_data", out_dir),
              help="Directory to put the training data. [default %default]"),
  make_option(c("--dropout"), type="double", default=0.5,
              help="Keep probability for training dropout. [default %default]"),
  make_option(c("--lr_anneal_step"), type="integer", default=10000,
              help="The epoch's step to change learning rate. [default %default]"),
  make_option(c("--network_type"), type="character", default="mlp",
              help="The network type to use. [default %default]"),
  make_option(c("--data_features_file"), type="character", default=ul_reduced_prdata_file,
              help="The Rdata file with features matrix. [default %default]"),
  make_option(c("--data_targets_file"), type="character", default=users_prdata_file,
              help="The Rdata file with ground truth dependent variables per sample in features matrix. [default %default]")
)
parser <- OptionParser(usage = "%prog [options] file", option_list = option_list, add_help_option = TRUE, 
                       description = "This is Fully Connected Feed Forward Deep Learning Network model around Tensorflow")
args <- parse_args(parser, positional_arguments = TRUE)
FLAGS <- args$options

# load script with appropriate network type
s_file <- sprintf("src/%s.R", FLAGS$network_type)
source(s_file)

# set random number generator's seed - so results will be stable from run to run
set.seed(44)

# the statring learning rate
learning_rate <- FLAGS$learning_rate


# Generate placeholder variables to represent the input tensors.
#
# These placeholders are used as inputs by the rest of the model building
# code and will be fed from the downloaded data in the .run() loop, below.
#
# Args:
#   batch_size: The batch size will be baked into both placeholders.
#
# Returns:
#   placeholders$features: the Users-Likes placeholder.
#   placeholders$labels: the user traits placeholder.
#   placeholders$keep_prob: the dropout keep probability
#
placeholder_inputs <- function(batch_size, features_dimensions) {
  
  # Note that the shapes of the placeholders match the shapes of the full
  # User-Likes and user traits tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  features <- tf$placeholder(tf$float32, shape(batch_size, features_dimensions))
  user_traits <- tf$placeholder(tf$float32, shape(batch_size, OUTPUTS_DIMENSION))
  keep_prob <- tf$placeholder(tf$float32)
  
  # return both placeholders
  list(features = features, labels = user_traits, keep_prob = keep_prob)
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
#   data_set: The set of user likes and user traits, from ul.read_data_sets()
#   placeholders: The list of parameters and inputs placeholders, from placeholder_inputs().
#   train: the flag to indicate if train data set generated
#
# Returns:
#   feed_dict: The feed dictionary mapping from placeholders to values.
#
fill_feed_dict <- function(data_set, placeholders, train) {
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  batch <- data_set$next_batch(FLAGS$batch_size)
  users_preprocessed = batch$users[,-1] # removing userid - it has unique value
  #users_preprocessed['age'] <- scale(batch$users['age']) # scale and center AGE
  
  # Convert data sets to matrix
  t_users <- as.matrix(users_preprocessed)
  t_ul <- as.matrix(batch$users_likes)
  if (train) {
    keep_prob = FLAGS$dropout
  } else {
    keep_prob = 1.0 # No dropout during testing
  }
  features_pl = placeholders$features
  labels_pl = placeholders$labels
  keep_prob_pl = placeholders$keep_prob
  dict(
    features_pl = t_ul,
    labels_pl = t_users,
    keep_prob_pl = keep_prob
  )
}

# Runs one evaluation against the full epoch of data.
#
# Args:
#   sess: The session in which the model has been trained.
#   predict_op: The Tensor that returns model predictions.
#   placeholders: The list of parameters and inputs placeholders, from placeholder_inputs().
#   data_set: The set of features and labels to evaluate,
#             from input_data.read_data_sets().
#   train: if set to TRUE then evaluation for train data
#
do_eval <- function(sess,
                    predict_op,
                    placeholders,
                    data_set, train) {
  # And run one epoch of eval.
  steps_per_epoch <- data_set$num_examples %/% FLAGS$batch_size
  num_examples <- steps_per_epoch * FLAGS$batch_size
  # The collected predictions vs labes per step
  predictions <- data.frame()
  labels <- data.frame()
  # Try to go over all data examples (approximatelly, at least taking the same number of batches as during training) 
  # and evaluate accuracy per step (batch)
  for (step in 1:steps_per_epoch) {
    feed_dict <- fill_feed_dict(data_set, placeholders, train)
    # Do predictions
    predicted <- sess$run(predict_op, feed_dict = feed_dict)
    predictions <- rbind(predictions, predicted)
    labels <- rbind(labels, feed_dict$items()[[2]][[2]])
  }
  
  # show summary of results
  vars <- data_set$labels_names[-1]
  accuracies <- c()
  cat("Prediction accuracies:\n")
  for(i in 1:OUTPUTS_DIMENSION) {
    # find accuracies per column
    accuracies[i] <- accuracy(labels[,i], predictions[,i])[[1]]
    # cat(sprintf("%9s : %.2f%%\n", vars[i], (accuracies[i] * 100.0)), file = regr_pred_accuracy_file, append = TRUE)
    cat(sprintf("%9s : %.2f%%\n", vars[i], (accuracies[i] * 100.0))) # to console
  }
  cat(sprintf("------------------\n"))
  cat(sprintf("%9s : %.2f%%\n", "Mean", .rowMeans(accuracies, 1, OUTPUTS_DIMENSION) * 100.0)) # to console
  cat(sprintf("%9s : %.2f%%\n", "Std", sd(accuracies) * 100.0)) # to console
  # Calculate loss
  err <- as.matrix(predictions - labels)
  mse <- mean(err ^ 2) # MSE
  mae <- mean(abs(err))
  cat(sprintf("Evaluation MSE: %.2f, MAE: %.2f\n", mse, mae))
}

#
# Train users psychometric model
#
tf$logging$set_verbosity(verbosity = tf$logging$DEBUG)

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(FLAGS$data_targets_file))
assertthat::assert_that(file.exists(FLAGS$data_features_file))

# Get sets of users-likes and users traits for train and test
data_sets <- ul_read_data_set(ul_file = FLAGS$data_features_file, users_file = FLAGS$data_targets_file)

# List to store train and test errors per step
errors <- list(train = c(), test = c())

# The units per layer
layers <- as.integer(strsplit(FLAGS$layers, ",")[[1]])
print(sprintf("Building NN with layers: [%s]", FLAGS$layers))

# Tell TensorFlow that the model will be built into the default Graph.
with(tf$Graph()$as_default(), {
  # Generate placeholders for the users-likes and users
  placeholders <- placeholder_inputs(FLAGS$batch_size, data_sets$features_dimension)
  
  # Build a Graph that computes predictions from the inference model.
  predicts <- inference(placeholders$features, layers, placeholders$keep_prob)
  
  # Add to the Graph the Ops for training loss calculation.
  loss_op <- loss(predicts, placeholders$labels)
  
  # Add to the Graph the Ops that calculate and apply gradients.
  train_op <- training(loss_op, FLAGS$learning_rate, FLAGS$lr_anneal_step)
  
  # Summarise NN biases and weights
  tf$contrib$layers$summarize_biases()
  tf$contrib$layers$summarize_weights()
  tf$contrib$layers$summarize_variables()

  # Build the summary Tensor based on the TF collection of Summaries.
  summary <- tf$summary$merge_all()

  # Add the variable initializer Op.
  init <- tf$global_variables_initializer()
  
  # Create a saver for writing training checkpoints.
  saver <- tf$train$Saver()
  
  # Create a session for running Ops on the Graph.
  sess <- tf$Session()
  
  # Instantiate a SummaryWriter to output summaries and the Graph.
  session_summary_dir <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
  session_summary_dir <- sprintf("%s/%s", FLAGS$train_dir, session_summary_dir)
  summary_writer_train <- tf$summary$FileWriter(sprintf("%s/%s", session_summary_dir, "train"), sess$graph)
  summary_writer_test <- tf$summary$FileWriter(sprintf("%s/%s", session_summary_dir, "test"), sess$graph)
  
  # And then after everything is built:
  
  # Run the Op to initialize the variables.
  sess$run(init)
  
  # Start the training loop.
  for (step in 1:FLAGS$max_steps) {
    start_time <- Sys.time()

    # Fill a feed dictionary with the actual set of users-likes and users
    # for this particular training step.
    feed_dict <- fill_feed_dict(data_set = data_sets$train,
                                placeholders, train = TRUE)
    
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    values <- sess$run(list(summary, train_op, loss_op), feed_dict = feed_dict)
    summary_str_train <- values[[1]]
    train_loss_value <- values[[3]]
    errors$train <- append(errors$train, train_loss_value)
    
    # The duration of train step
    duration <- Sys.time() - start_time
    
    # Write the summaries and print an overview fairly often.
    if (step %% 100 == 0) {
      # Update the train events file.
      summary_writer_train$add_summary(summary_str_train, step)

      # Calculate loss over test data
      test_feed_dict <- fill_feed_dict(data_set = data_sets$test,
                                       placeholders, train = FALSE)
      test_values <- sess$run(list(summary, loss_op), feed_dict = test_feed_dict)
      summary_str_test <- test_values[[1]]
      test_loss_value <- test_values[[2]]
      errors$test <- append(errors$test, test_loss_value)
      # Update the test events file
      summary_writer_test$add_summary(summary_str_test, step)
      
      # Print status to stdout.
      cat(sprintf('Step %d: train loss = %.2f, test loss = %.2f (duration: %s)\n',
                  step, train_loss_value, test_loss_value, duration))
      
      # Flush summaries
      summary_writer_train$flush()
      summary_writer_test$flush()
    }
    
    # Save a checkpoint and evaluate the model periodically.
    if ((step + 1) %% 1000 == 0 || (step + 1) == FLAGS$max_steps) {
      checkpoint_file <- file.path(session_summary_dir, 'checkpoint')
      saver$save(sess, checkpoint_file, global_step = step)
      
      # Evaluate against the training set.
      cat('\nTraining Data Eval:\n')
      do_eval(sess,
              predicts,
              placeholders,
              data_sets$train, 
              train = TRUE)
      
      # Evaluate against the test set.
      cat('Test Data Eval:\n')
      do_eval(sess,
              predicts,
              placeholders,
              data_sets$test, 
              train = FALSE)
    } 
  }
  
  # Final details about method
  cat(sprintf("Learning rate start: %g, dropout = %.2f, input_features = %d, layers = [%s]\n",
              FLAGS$learning_rate, FLAGS$dropout, data_sets$features_dimension, FLAGS$layers))
  train_error <- mean(errors$train)
  test_error <- mean(errors$test)
  cat(sprintf("Mean train/test errors: %.4f / %.4f, train optimizer: %s\n", train_error, test_error, train_op$name))
})






