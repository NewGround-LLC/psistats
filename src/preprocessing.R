# This is preprocessing routines to convert provided CVS data files into
# appropriate sparse matrix. As part of the process some data will be trimmed
# by removing rare users and likes from the data (see trimMatrix).
#
# Make sure to set working directory appropriately - all path related
#
source('./src/utils.R')

# The preprocessing parameters
#
minUsersPerLike <- 150 # the minimum number of users per like to keep like in the data (20)
minLikesPerUser <- 50 # the minimum number of likes per user to keep user in the data (2)

# Loading all related data sets
#
users <- read.csv("DataSets/sample_dataset/users.csv")
cat("Users:\t\t", dim(users), "\n")

likes <- read.csv("DataSets/sample_dataset/likes.csv")
cat("Likes:\t\t", dim(likes), "\n")

ul <- read.csv("DataSets/sample_dataset/users-likes.csv")
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
M <- trimMatrix(M, minUsersPerLike, minLikesPerUser)
print("\nTrimmed matrix\n")
printULSummary(M)

# Remove users deleted in M from users object
users <- users[match(rownames(M), users$userid),]

# save intermediates
save(M, file = "out/intermediate/M.RData")
save(users, file = "out/intermediate/users.RData")
