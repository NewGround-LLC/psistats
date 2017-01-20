source('./src/utils.R')

# Make sure to set working directory - all path related
#
# Loading all related data sets
#
users <- read.csv("DataSets/sample_dataset/users.csv")
likes <- read.csv("DataSets/sample_dataset/likes.csv")
ul <- read.csv("DataSets/sample_dataset/users-likes.csv")
dim(users)
dim(likes)
dim(ul)

# Constructing a User-Like Matrix
#
ul$user_row <- match(ul$userid, users$userid)
ul$like_row <- match(ul$likeid, likes$likeid)

require(Matrix)
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x=1)
rownames(M) <- users$userid
colnames(M) <- likes$name

# Print matrix
printULSummary(M)

# Now remove obsolete objects to free memory
rm(ul, likes)

# Trimming the user-footprint matrix
#
M <- trimMatrix(M, 150, 50)
printULSummary(M)

# Remove users deleted in M from users object
users <- users[match(rownames(M), users$userid),]
