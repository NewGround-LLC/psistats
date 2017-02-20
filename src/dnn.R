# The Deep Learning Network configuration
# 
# Implements the inference/loss/training pattern for model building.
# 
# 1. inference() - Builds the model as far as is required for running the network
# forward to make predictions.
# 2. loss() - Adds to the inference model the layers required to generate loss.
# 3. training() - Adds to the loss model the Ops required to generate and
# apply gradients.
library(tensorflow)

# The number of output dimensions
OUTPUTS_DIMENSION <- 8L

# Build the psyhodemographic data model up to where it may be used for inference.
#
# Args:
#   features: Users-Likes placeholder, from placeholder_inputs().
#   layers: The vector with number of units per layer.
#   keep_prob: dropout probability placeholder, from placeholder_inputs().
#
# Returns:
#   softmax_linear: Output tensor with the computed logits.
#
inference <- function(features, layers, keep_prob) {
  
  # The dropout function
  dropout <- function(input_tensor, keep_prob_tensor, layer_name) {
    with(tf$name_scope(layer_name), {
      tf$summary$scalar("dropout_keep_probability", keep_prob_tensor)
      dropped <- tf$nn$dropout(input_tensor, keep_prob_tensor)
    })
    dropped
  }
  
  # Hidden 1
  features_dimension <- features$get_shape()$as_list()[2]
  hidden1 <- tf$contrib$layers$fully_connected(inputs = features, 
                                               num_outputs = layers[1], 
                                               activation_fn = tf$nn$relu,
                                               weights_initializer = tf$contrib$layers$xavier_initializer(),
                                               weights_regularizer = tf$contrib$layers$l2_regularizer(0.0001),
                                               biases_initializer = tf$contrib$layers$xavier_initializer())
  tf$contrib$layers$summarize_activation(hidden1)
  # Apply dropout to avoid model overfitting on training data
  dropped1 <- dropout(hidden1, keep_prob, "dropout_hidden1")
  
  # The linear regression layer
  # linear = tf$contrib$layers$linear(inputs = dropped1, num_outputs = layers[2])
  
  # Hidden 2
  hidden2 <- tf$contrib$layers$fully_connected(inputs = dropped1, 
                                               num_outputs = layers[2], 
                                               activation_fn = tf$nn$relu,
                                               weights_initializer = tf$contrib$layers$xavier_initializer(),
                                               weights_regularizer = tf$contrib$layers$l2_regularizer(0.0001),
                                               biases_initializer = tf$contrib$layers$xavier_initializer())
  tf$contrib$layers$summarize_activation(hidden2)
  # Apply dropout to avoid model overfitting on training data
  dropped2 <- dropout(hidden2, keep_prob, "dropout_hidden2")
  
  # Return linear regression output layer
  out = tf$contrib$layers$linear(inputs = dropped2, num_outputs = OUTPUTS_DIMENSION)
}

# Calculates prediction error from the predictions and the ground truth
#
# Args:
#   predicts: predictions tensor, float - [batch_size, OUTPUTS_DIMENSION].
#   gt_labels: the ground truth labels tensor, float - [batch_size, OUTPUTS_DIMENSION].
#
# Returns:
#   loss: prediction error tensor of type float.
prediction_error <- function(predicts, gt_labels) {
  err <- (predicts - gt_labels) ^ 2 # Squared Error
  #err <- tf$abs(predicts - gt_labels) # Absolute Error
  with(tf$name_scope("total"), {
    loss <- tf$reduce_mean(err) # Mean Squared(Absolute) Error
    tf$summary$scalar("loss", loss)
  })
  loss
}

# Calculates the train loss from the predictions and the ground truth
#
# Args:
#   predicts: predictions tensor, float - [batch_size, OUTPUTS_DIMENSION].
#   gt_labels: the ground truth labels tensor, float - [batch_size, OUTPUTS_DIMENSION].
#
# Returns:
#   loss: Loss tensor of type float.
#
loss <- function(predicts, gt_labels) {
  with(tf$name_scope("Loss"), {
    loss <- prediction_error(predicts, gt_labels)
  })
  loss
}

# Sets up the training Ops.
#
# Creates a summarizer to track the loss over time in TensorBoard.
#
# Creates an optimizer and applies the gradients to all trainable variables.
#
# The Op returned by this function is what must be passed to the
# `sess.run()` call to cause the model to train.
#
# Args:
#   loss: Loss tensor, from loss().
#   learning_rate_start: The starting learning rate to use for gradient descent.
#   lr_anneal_step: The decay steps for learning rate
#   lr_decay_rate: The learning rate decay rate
#
# Returns:
#   train_op: The Op for training.
#
training <- function(loss, learning_rate_start, lr_anneal_step, lr_decay_rate = 0.96) {
  with(tf$name_scope("train"), {
    # Create a variable to track the global step.
    global_step <- tf$Variable(0L, name = 'global_step', trainable = FALSE)
    learning_rate = tf$train$exponential_decay(learning_rate = learning_rate_start, global_step = global_step, 
                                               decay_steps = lr_anneal_step, decay_rate = lr_decay_rate, 
                                               staircase=TRUE)
    
    # Add a scalar summary for the snapshot loss.
    tf$summary$scalar(loss$op$name, loss)
    # Add learning rate to summary to comapare with different learning rates
    tf$summary$scalar("learning_rate", learning_rate)
    
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer <- tf$train$GradientDescentOptimizer(learning_rate)
    #optimizer <- tf$train$AdagradOptimizer(learning_rate = learning_rate)
    optimizer <- tf$train$AdamOptimizer(learning_rate = learning_rate)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op <- optimizer$minimize(loss, global_step = global_step)
  })
  train_op
}