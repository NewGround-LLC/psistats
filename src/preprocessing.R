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
  make_option(c("-u", "--min_users"), type="integer", default=150,
              help="the minimum number of users per like to keep like in the data [default %default]"),
  make_option(c("-l", "--min_likes"), type="integer", default=50,
              help="the minimum number of likes per user to keep user in the data [default %default]")
)
parser <- OptionParser(usage = "%prog [options] file", option_list = option_list, add_help_option = TRUE, 
                       description = "This is preprocessing routines to convert provided CVS data files into appropriate sparse matrix. As part of the process some data will be trimmed by removing rare users and likes from the data.")
args <- parse_args(parser, positional_arguments = TRUE)
opt <- args$options

print(sprintf("Min users per like: %d, min likes per user: %d", opt$min_users, opt$min_likes))

# Check that input data exist
print("Checking that input data files exist")
assertthat::assert_that(file.exists(input_users_csv))
assertthat::assert_that(file.exists(input_likes_csv))
assertthat::assert_that(file.exists(input_ul_csv))

# Loading all related data sets
#
users <- read.csv(input_users_csv)
cat(sprintf("%12s : [%d, %d]\n","Users", dim(users)[1], dim(users)[2]))

likes <- read.csv(input_likes_csv)
cat(sprintf("%12s : [%d, %d]\n","Likes", dim(likes)[1], dim(likes)[2]))

ul <- read.csv(input_ul_csv)
cat(sprintf("%12s : [%d, %d]\n","Users-Likes", dim(ul)[1], dim(ul)[2]))

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

# The 'Political' dependent variable has multiple missing values. Do apply multiple imputation to the dependent variables
# in order to substitude missed values with predicted
print("Impute missed values in users")
library(mice)
users_imp <- users[,2:9] # omit userid
users_imp[,3] <- factor(users_imp[,3]) # factorize Political
imp <- mice(users_imp, m = 5, seed = 42, method = c("", "", "lda", "", "", "", "", ""))
users_complete <- complete(imp) # fill in missing data
users_complete[,3] <- as.integer(users_complete[,3]) - 1 # to get back "0", "1"
users[,2:9] <- users_complete # assign to original data

print("Imputation results summary")
fit <- with(imp, lm(as.integer(political) ~ gender + age + ope + con + ext + agr + neu))
round(summary(pool(fit)), 2)

# save intermediates
if (!dir.exists(out_intermediates_dir))
  dir.create(out_intermediates_dir, recursive = TRUE)

save(M, file = ul_prdata_file)
save(users, file = users_prdata_file)
