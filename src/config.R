# The global configuration holder
#
# The number of folds
n_folds <- 10 
# The number of SVD dimensions
K <- 50

#
# The directory to load data sets from
input_data_dir <- "DataSets/sample_dataset"
# The input users CSV data file
input_users_csv <- sprintf("%s/users.csv", input_data_dir)
# The input likes CSV data file
input_likes_csv <- sprintf("%s/likes.csv", input_data_dir)
# The input users-likes CSV data file
input_ul_csv <- sprintf("%s/users-likes.csv", input_data_dir)

#
# The directory to hold outputs
out_dir <- "out"
# The directory to hold preprocessed outputs
out_intermediates_dir <- sprintf("%s/intermediate", out_dir)

# The intermediate file to hold preprocessed users data
users_prdata_file <- sprintf("%s/users.RData", out_intermediates_dir)
# The intermediate file to store preprocessed Users-Likes sparse matrix
ul_prdata_file <- sprintf("%s/M.RData", out_intermediates_dir)
# The intermediate file to store preprocessed Users-Likes sparse matrix with features dimensions reduced
ul_reduced_prdata_file <- sprintf("%s/M_reduced.RData", out_intermediates_dir)

# The output file to store prediction accuracy for regression analysis
regr_pred_accuracy_file <- sprintf("%s/pred_accuracy_regr.txt", out_dir)