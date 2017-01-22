# The global configuration holder
#

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
# The intermediate file to store varimax rotated SVD scores of Users
users_rot_varimax_file <- sprintf("%s/usersSVDrot.RData", out_intermediates_dir)
# The cross-validation folds
users_cv_folds_file <- sprintf("%s/folds.RData", out_intermediates_dir)
