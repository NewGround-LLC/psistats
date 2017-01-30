# Builds psyhodemographic data deep learning network
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
OUTPUTS_DIMENSION <- 8

# Build the psyhodemographic data model up to where it may be used for inference.
#
# Args:
#   features: Users-Likes placeholder, from inputs().
#   hidden1_units: Size of the first hidden layer.
#   hidden2_units: Size of the second hidden layer.
#
# Returns:
#   softmax_linear: Output tensor with the computed logits.
#
inference <- function(features, hidden1_units, hidden2_units) {
  # We can't initialize these variables to 0 - the network will get stuck.
  weight_variable <- function(shape) {
    initial <- tf$truncated_normal(shape, stddev = 0.1 / sqrt(shape[[2]]))
    tf$Variable(initial)
  }
  
  bias_variable <- function(shape) {
    initial <- tf$constant(0.1, shape = shape)
    tf$Variable(initial)
  }
  
  # Attach a lot of summaries to a Tensor
  variable_summaries <- function(var, name) {
    with(tf$name_scope("summaries"), {
      mean <- tf$reduce_mean(var)
      tf$summary$scalar(paste0("mean/", name), mean)
      with(tf$name_scope("stddev"), {
        stddev <- tf$sqrt(tf$reduce_mean(tf$square(var - mean)))
      })
      tf$summary$scalar(paste0("stddev/", name), stddev)
      tf$summary$scalar(paste0("max/", name), tf$reduce_max(var))
      tf$summary$scalar(paste0("min/", name), tf$reduce_min(var))
      tf$summary$histogram(name, var)
    })
  }
  
  # Reusable code for making a simple neural net layer.
  #
  # It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  # It also sets up name scoping so that the resultant graph is easy to read,
  # and adds a number of summary ops.
  #
  nn_layer <- function(input_tensor, input_dim, output_dim,
                       layer_name, act=tf$nn$relu) {
    with(tf$name_scope(layer_name), {
      # This Variable will hold the state of the weights for the layer
      with(tf$name_scope("weights"), {
        weights <- weight_variable(shape(input_dim, output_dim))
        variable_summaries(weights, paste0(layer_name, "/weights"))
      })
      with(tf$name_scope("biases"), {
        biases <- bias_variable(shape(output_dim))
        variable_summaries(biases, paste0(layer_name, "/biases"))
      })
      with (tf$name_scope("Wx_plus_b"), {
        preactivate <- tf$matmul(input_tensor, weights) + biases
        tf$summary$histogram(paste0(layer_name, "/pre_activations"), preactivate)
      })
      activations <- act(preactivate, name = "activation")
      tf$summary$histogram(paste0(layer_name, "/activations"), activations)
    })
    activations
  }
  
  # The no operation function
  noop <- function(input_tensor, name) {
    input_tensor # just return input
  }
  
  # Hidden 1
  features_dimension <- features$get_shape()$as_list()[2]
  hidden1 <- nn_layer(input_tensor = features, features_dimension, hidden1_units, "hidden1")
  
  # Hidden 2
  hidden2 <- nn_layer(input_tensor = hidden1, hidden1_units, hidden2_units, "hidden2")
  
  # Return linear regression output layer
  nn_layer(input_tensor = hidden2, hidden2_units, OUTPUTS_DIMENSION, "linear", act = noop)
}

# Calculates the loss from the predictions and the ground truth
#
# Args:
#   predicts: predictions tensor, float - [batch_size, OUTPUTS_DIMENSION].
#   gt_labels: the ground truth labels tensor, float - [batch_size, OUTPUTS_DIMENSION].
#
# Returns:
#   loss: Loss tensor of type float.
#
loss <- function(predicts, gt_labels) {
  with(tf$name_scope("MSE"), {
    mse <- (predicts - gt_labels) ^ 2
    with(tf$name_scope("total"), {
      loss <- tf$reduce_mean(mse) # MSE
    })
    tf$summary$scalar("MSE", loss)
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
#   learning_rate: The learning rate to use for gradient descent.
#
# Returns:
#   train_op: The Op for training.
#
training <- function(loss, learning_rate) {
  with(tf$name_scope("train"), {
    # Add a scalar summary for the snapshot loss.
    tf$summary$scalar(loss$op$name, loss)
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer <- tf$train$GradientDescentOptimizer(learning_rate)
    
    # Create a variable to track the global step.
    global_step <- tf$Variable(0L, name = 'global_step', trainable = FALSE)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op <- optimizer$minimize(loss, global_step = global_step)
  })
  train_op
}