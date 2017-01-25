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
#   placeholders$users_likes: the Users-Likes placeholder.
#   placeholders$users: the user traits placeholder.
#
placeholder_inputs <- function(batch_size, features_dimensions) {
  
  # Note that the shapes of the placeholders match the shapes of the full
  # User-Likes and user traits tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  users_likes <- tf$placeholder(tf$float32, shape(batch_size, features_dimensions))
  user_traits <- tf$placeholder(tf$int32, shape(batch_size, OUTPUTS_DIMENSION))
  
  # return both placeholders
  list(users_likes = users_likes, users = user_traits)
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
#   users_likes_pl: the Users-Likes placeholder, from placeholder_inputs().
#   users_pl: the users traits placeholder, from placeholder_inputs().
#
# Returns:
#   feed_dict: The feed dictionary mapping from placeholders to values.
#
fill_feed_dict <- function(data_set, users_likes_pl, users_pl) {
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  batch <- data_set$next_batch(FLAGS$batch_size)
  users_preprocessed = batch$users[,-1] # removing userid - it has unique value
  users_preprocessed['age'] <- scale(batch$users['age']) # scale and center AGE to avoid bias in RELU
  dict(
    users_likes_pl = batch$users_likes,
    users_pl = users_preprocessed
  )
}

# Train users psychometric model

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(users_prdata_file))
assertthat::assert_that(file.exists(ul_prdata_file))

# Get sets of users-likes and users traits for train and test
data_sets <- ul.read_data_set(ul_file = ul_prdata_file, users_file = users_prdata_file)

# Tell TensorFlow that the model will be built into the default Graph.
with(tf$Graph()$as_default(), {
  
  # Generate placeholders for the users-likes and users
  placeholders <- placeholder_inputs(FLAGS$batch_size, data_sets$features_dimension)
  
  # Build a Graph that computes predictions from the inference model.
  predicts <- inference(placeholders$users_likes, FLAGS$hidden1, FLAGS$hidden2)
  
  # Add to the Graph the Ops for loss calculation.
  loss <- loss(predicts, placeholders$users)
  
  # Add to the Graph the Ops that calculate and apply gradients.
  train_op <- training(loss, FLAGS$learning_rate)
  
  # Add the Op to compare the predictions to the ground truth during evaluation.
  #eval_correct <- evaluation(predicts, placeholders$users)
  
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
                                users_likes_pl = placeholders$users_likes,
                                users_pl = placeholders$users)
    
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
      cat(sprintf('Step %d: loss = %.2f (%.3f sec)\n',
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
      
    }
  }
})






