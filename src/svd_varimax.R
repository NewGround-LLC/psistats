# This is dimensionality reduction routines aimed to reduce number of features dimensions
# in the data corpus using SVD with varimax factor rotation analysis.
source('./src/users_likes_data_set.R')

library(optparse)

# parse command line arguments
option_list <- list(
  make_option(c("-k", "--svd_dimensions"), type="integer", default=50,
              help="the number of SVD dimensions to apply when doing features dimensionality reduction [default %default]"),
  make_option(c("-v", "--apply_varimax"), type="logical", default=TRUE,
              help="the flag to indicate whether varimax rotation should be applied to results of SVD factor analysis [default %default]")
)
parser <- OptionParser(usage = "%prog [options] file", option_list = option_list, add_help_option = TRUE, 
                       description = "This will reduce number of features dimensions by applying SVD.")
args <- parse_args(parser, positional_arguments = TRUE)
opt <- args$options
print(sprintf("Number of SVD dimensions to apply: %d, apply varimax: %s", opt$svd_dimensions, opt$apply_varimax))

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(ul_prdata_file))

# run SVD 
out_file <- sprintf("%s/M_reduced_%d.RData", out_intermediates_dir, opt$svd_dimensions)
ul_save_features_reduced_data_set(ul_file = ul_prdata_file, out_file = out_file, svd_k = opt$svd_dimensions, varimax_rotate = opt$apply_varimax)