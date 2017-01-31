# This is analysis based on Fully Connected Feed Forward Deep Learning Network build with TensorFlow framework.

source('./src/config.R')
source('./src/dnn.R')
source('./src/users_likes_data_set.R')
source('./src/utils.R')

# make sure that library installed
library(tensorflow)
library(optparse)

# Basic model parameters as external flags.
option_list <- list(
  make_option(c("--learning_rate"), type="double", default=0.05,
              help="Initial learning rate. [default %default]"),
  make_option(c("--max_steps"), type="integer", default=50000L,
              help="Number of steps to run trainer. [default %default]"),
  make_option(c("--hidden1"), type="integer", default=256L,
              help="Number of units in hidden layer 1. [default %default]"),
  make_option(c("--hidden2"), type="integer", default=64L,
              help="Number of units in hidden layer 2. [default %default]"),
  make_option(c("--batch_size"), type="integer", default=100L,
              help="Batch size. Must divide evenly into the dataset sizes. [default %default]"),
  make_option(c("--train_dir"), type="character", default=sprintf("%s/train_data", out_dir),
              help="Directory to put the training data. [default %default]")
)
parser <- OptionParser(usage = "%prog [options] file", option_list = option_list, add_help_option = TRUE, 
                       description = "This is Fully Connected Feed Forward Deep Learning Network model around Tensorflow")
args <- parse_args(parser, positional_arguments = TRUE)
FLAGS <- args$options

# set random number generator's seed - so results will be stable from run to run
set.seed(44)


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
#
placeholder_inputs <- function(batch_size, features_dimensions) {
  
  # Note that the shapes of the placeholders match the shapes of the full
  # User-Likes and user traits tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  features <- tf$placeholder(tf$float32, shape(batch_size, features_dimensions))
  user_traits <- tf$placeholder(tf$float32, shape(batch_size, OUTPUTS_DIMENSION))
  
  # return both placeholders
  list(features = features, labels = user_traits)
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
#   features_pl: the Users-Likes placeholder, from placeholder_inputs().
#   labels_pl: the users traits placeholder, from placeholder_inputs().
#
# Returns:
#   feed_dict: The feed dictionary mapping from placeholders to values.
#
fill_feed_dict <- function(data_set, features_pl, labels_pl) {
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  batch <- data_set$next_batch(FLAGS$batch_size)
  users_preprocessed = batch$users[,-1] # removing userid - it has unique value
  #users_preprocessed['age'] <- scale(batch$users['age']) # scale and center AGE
  
  # Convert data sets to matrix
  t_users <- as.matrix(users_preprocessed)
  t_ul <- as.matrix(batch$users_likes)
  dict(
    features_pl = t_ul,
    labels_pl = t_users
  )
}

# Runs one evaluation against the full epoch of data.
#
# Args:
#   sess: The session in which the model has been trained.
#   predict_op: The Tensor that returns model predictions.
#   features_placeholder: The features placeholder.
#   labels_placeholder: The labels placeholder.
#   data_set: The set of features and labels to evaluate,
#             from input_data.read_data_sets().
#
do_eval <- function(sess,
                    predict_op,
                    features_placeholder,
                    labels_placeholder,
                    data_set) {
  # And run one epoch of eval.
  steps_per_epoch <- data_set$num_examples %/% FLAGS$batch_size
  num_examples <- steps_per_epoch * FLAGS$batch_size
  # The collected predictions vs labes per step
  predictions <- data.frame()
  labels <- data.frame()
  # Try to go over all data examples (approximatelly, at least taking the same number of batches as during training) 
  # and evaluate accuracy per step (batch)
  for (step in 1:steps_per_epoch) {
    feed_dict <- fill_feed_dict(data_set,
                                features_placeholder,
                                labels_placeholder)
    predicted <- sess$run(predict_op, feed_dict = feed_dict)
    predictions <- rbind(predictions, predicted)
    labels <- rbind(labels, feed_dict$items()[[2]][[2]])
  }
  
  # show summary of results
  vars <- data_set$labels_names[-1]
  accuracies <- list()
  cat("Prediction accuracies:\n")
  for(i in 1:OUTPUTS_DIMENSION) {
    # fix predictions
    nan_ind <- which(predictions[,i] == 'NaN')
    predictions[,i][nan_ind] <- NA
    
    # find accuracies per column
    accuracies[[i]] = accuracy(labels[,i], predictions[,i])
    # cat(sprintf("%9s : %.2f%%\n", vars[i], (accuracies[[i]][[1]] * 100.0)), file = regr_pred_accuracy_file, append = TRUE)
    cat(sprintf("%9s : %.2f%%\n", vars[i], (accuracies[[i]][[1]] * 100.0))) # to console
  }
}

#
# Train users psychometric model
#
tf$logging$set_verbosity(verbosity = tf$logging$DEBUG)

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_reduced_prdata_file))

# Get sets of users-likes and users traits for train and test
data_sets <- ul_read_data_set(ul_file = ul_reduced_prdata_file, users_file = users_prdata_file)

# Tell TensorFlow that the model will be built into the default Graph.
with(tf$Graph()$as_default(), {
  
  # Generate placeholders for the users-likes and users
  placeholders <- placeholder_inputs(FLAGS$batch_size, data_sets$features_dimension)
  
  # Build a Graph that computes predictions from the inference model.
  predicts <- inference(placeholders$features, FLAGS$hidden1, FLAGS$hidden2)
  
  # Add to the Graph the Ops for loss calculation.
  loss <- loss(predicts, placeholders$labels)
  
  # Add to the Graph the Ops that calculate and apply gradients.
  train_op <- training(loss, FLAGS$learning_rate)
  
  # Add the Op to compare the predictions to the ground truth during evaluation.
  # eval_correct <- evaluation(predicts, placeholders$labels)
  
  # Build the summary Tensor based on the TF collection of Summaries.
  summary <- tf$summary$merge_all()
  
  # Add the variable initializer Op.
  init <- tf$global_variables_initializer()
  
  # Create a saver for writing training checkpoints.
  saver <- tf$train$Saver()
  
  # Create a session for running Ops on the Graph.
  sess <- tf$Session()
  
  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer <- tf$summary$FileWriter(FLAGS$train_dir, sess$graph)
  
  # And then after everything is built:
  
  # Run the Op to initialize the variables.
  sess$run(init)
  
  # Start the training loop.
  for (step in 1:FLAGS$max_steps) {
    start_time <- Sys.time()
    
    # Fill a feed dictionary with the actual set of users-likes and users
    # for this particular training step.
    feed_dict <- fill_feed_dict(data_set = data_sets$train,
                                features_pl = placeholders$features,
                                labels_pl = placeholders$labels)
    
    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    values <- sess$run(list(train_op, loss), feed_dict = feed_dict)
    loss_value <- values[[2]]
    
    duration <- Sys.time() - start_time
    
    # Write the summaries and print an overview fairly often.
    if (step %% 100 == 0) {
      # Print status to stdout.
      cat(sprintf('Step %d: loss = %.2f (%s)\n',
                  step, loss_value, duration))
      # Update the events file.
      summary_str <- sess$run(summary, feed_dict = feed_dict)
      summary_writer$add_summary(summary_str, step)
      summary_writer$flush()
    }
    
    # Save a checkpoint and evaluate the model periodically.
    if ((step + 1) %% 1000 == 0 || (step + 1) == FLAGS$max_steps) {
      checkpoint_file <- file.path(FLAGS$train_dir, 'checkpoint')
      saver$save(sess, checkpoint_file, global_step = step)
      
      # Evaluate against the training set.
      cat('\nTraining Data Eval:\n')
      do_eval(sess,
              predicts,
              placeholders$features,
              placeholders$labels,
              data_sets$train)
      
      # Evaluate against the test set.
      cat('Test Data Eval:\n')
      do_eval(sess,
              predicts,
              placeholders$features,
              placeholders$labels,
              data_sets$test)
    } 
  }
  
  # Final details about method
  cat(sprintf("Learning rate: %.4f, input_features = %d, hidden1 = %d, hidden2 = %d",
              FLAGS$learning_rate, data_sets$features_dimension, FLAGS$hidden1, FLAGS$hidden2))
})






