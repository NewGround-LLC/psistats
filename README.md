This repository contains the source code accompanying research paper "Applying Deep Machine Learning for psycho-demographic profiling of Internet users using O.C.E.A.N. model of personality". 

# Setting up working environment

## Dependencies
The source code in this repository is  written in `R` programming language and use `TensorFlow` framework to accelerate graph computations performed by studied machine learning models. For better performance, it's recommended to install `TensorFlow` with GPU support.

The installation instructions for `TensorFlow` library may be found at [1]
The installation instructions for `R` programming language environment described in section _R Installation and Administration_ of R manuals at [2]

In order to get access from R environment to `TensorFlow` framework its necessary to install `TensorFlow for R` package as described at [3]


The R source code has multiple dependencies on third party R packages:
* irlba - the package to perform singular value decomposition analysis
* optparse - the package providing support for command line arguments parsing
* Matrix - the package for handling sparse matrix structures
* mice - the package to perform multivariate imputation of missing values
* R6 - the package allowing to define `R6Class` class objects in R
* ROCR - the package for visualizing the performance of scoring classifiers
* tensorflow - the package exposing `TensorFlow` Python API in `R` environment

In order to install necessary R packages run the following command in R environment:
```
> install.packages(c("irlba", "optparse", "Matrix", "mice", "R6", "ROCR"))
```
The `tensorflow` bridge package should be installed as described at [3]

## Data Corpus
The data corpus used in the research is publicly available and can be requested at [4] 

The data corpuse comprise of following files:

1. _users.csv:_ contains psycho-demographic user profiles. It has nu = 110 728 rows (excluding the row holding column names) and nine columns: anonymized user ID, gender (“0” for male and “1” for female), age, political views (“0” for Democrat and “1” for Republican), and scores of five-factor model of personality (Goldberg et al., 2006).
2. _likes.csv:_ contains anonymized IDs and names of nL = 1 580 284 Facebook Likes. It has two columns: ID and name.
3. _users-likes.csv:_ contains the associations between users and their Likes, stored as user–Like pairs. It has nu-L = 10 612 326 rows and two columns: user ID and Like ID. An existence of a user–Like pair implies that a given user had the corresponding Like on their profile.


[1]:https://www.tensorflow.org/install/
[2]:https://cran.r-project.org/manuals.html
[3]:https://rstudio.github.io/tensorflow/
[4]:http://dataminingtutorial.com
