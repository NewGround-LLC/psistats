# This is preprocessing routines to convert provided CVS data files into
# appropriate sparse matrix. As part of the process some data will be trimmed
# by removing rare users and likes from the data (see trimMatrix).
#
# Make sure to set working directory appropriately - all path related
#
source('./src/utils.R')
source('./src/config.R')

library(optparse)

# parse command line arguments
option_list <- list(
  make_option(c("-mu", "--min_users"), type="integer", default=150,
              help="the minimum number of users per like to keep like in the data [default %default]"),
  make_option(c("-ml", "--min_likes"), type="integer", default=50,
              help="the minimum number of likes per user to keep user in the data [default %default]")
)
parser <- OptionParser(usage = "%prog [options] file", option_list = option_list, add_help_option = TRUE, 
                       description = "This is preprocessing routines to convert provided CVS data files into appropriate sparse matrix. As part of the process some data will be trimmed by removing rare users and likes from the data.")
args <- parse_args(parser, positional_arguments = TRUE)
opt <- args$options

# Loading all related data sets
#
users <- read.csv(input_users_csv)
cat("Users:\t\t", dim(users), "\n")

likes <- read.csv(input_likes_csv)
cat("Likes:\t\t", dim(likes), "\n")

ul <- read.csv(input_ul_csv)
cat("Users-Likes:\t", dim(ul), "\n")

# Constructing a User-Like Matrix
#
ul$user_row <- match(ul$userid, users$userid)
ul$like_row <- match(ul$likeid, likes$likeid)

require(Matrix)
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1)
rownames(M) <- users$userid
colnames(M) <- likes$name

# Print matrix
printULSummary(M)

# Now remove obsolete objects to free memory
rm(ul, likes)

# Trimming the user-footprint matrix
#
cat(sprintf("Trimmings matrix, minUsersPerLike: %d, minLikesPerUser: %d", opt$min_users, opt$min_likes))
M <- trimMatrix(M, opt$min_users, opt$min_likes)
cat("\nTrimmed matrix\n")
printULSummary(M)

# Remove users deleted in M from users object
users <- users[match(rownames(M), users$userid),]

# save intermediates
if (!dir.exists(out_intermediates_dir))
  dir.create(out_intermediates_dir, recursive = TRUE)

save(M, file = ul_prdata_file)
save(users, file = users_prdata_file)
