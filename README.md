This repository contains the source code accompanying research paper ["Applying Deep Machine Learning for psycho-demographic profiling of Internet users using O.C.E.A.N. model of personality"][5]. 

## Setting up working environment

### Dependencies
The source code in this repository is  written in `R` programming language and use `TensorFlow` framework to accelerate graph computations performed by studied machine learning models. For better performance, it's recommended to install `TensorFlow` with GPU support.

The installation instructions for `TensorFlow` library may be found at [this page][1].
The installation instructions for `R` programming language environment described in section _'R Installation and Administration'_ of manuals at [web page][2].

In order to get access from R environment to `TensorFlow` framework its necessary to install `TensorFlow for R` package as described at [this page][3]


The R source code has multiple dependencies on third party R packages:
* [irlba](https://www.rdocumentation.org/packages/irlba/versions/2.1.2/topics/irlba) - the package to perform singular value decomposition analysis
* [optparse](https://www.rdocumentation.org/packages/optparse/versions/1.3.2) - the package providing support for command line arguments parsing
* [Matrix](https://www.rdocumentation.org/packages/Matrix/versions/1.2-8) - the package for handling sparse matrix structures
* [mice](https://www.rdocumentation.org/packages/mice/versions/2.30) - the package to perform multivariate imputation of missing values
* [R6](https://www.rdocumentation.org/packages/R6/versions/2.2.0) - the package allowing to define `R6Class` classes with references semantics
* [ROCR](https://www.rdocumentation.org/packages/ROCR/versions/1.0-7) - the package for visualizing the performance of scoring classifiers
* [tensorflow](https://www.rdocumentation.org/packages/tensorflow/versions/0.6.0) - the package exposing `TensorFlow` Python API in `R` environment

In order to install necessary R packages run the following command in R environment:
```
> install.packages(c("irlba", "optparse", "Matrix", "mice", "R6", "ROCR"))
```
The `tensorflow` bridge package should be installed as described at [this manual][3]

### Data Corpus
The data corpus used in the research is publicly available and can be requested at [dataminingtutorial.com][4] 

The data corpus comprise of following files:

1. _users.csv:_ contains psycho-demographic user profiles. It has 110 728 rows (excluding the row holding column names) and nine columns: anonymized user ID, gender (“0” for male and “1” for female), age, political views (“0” for Democrat and “1” for Republican), and scores of five-factor model of personality (Goldberg et al., 2006).
2. _likes.csv:_ contains anonymized IDs and names of 1 580 284 Facebook Likes. It has two columns: ID and name.
3. _users-likes.csv:_ contains the associations between users and their Likes, stored as user–Like pairs. It has 10 612 326 rows and two columns: user ID and Like ID. An existence of a user–Like pair implies that a given user had the corresponding Like on their profile.


## Source code structure

The source code consist of `R` scripts, each encapsulating particular functionality:

* [config.R](src/config.R) - holds common configuration parameters (input, intermediate and output directories, etc)
* [preprocessing.R](src/preprocessing.R) - performs raw data corpus preprocessing by creating sparse data matrix, trimming it, and missing data point imputation
* [analysis.R](src/analysis.R) - encapsulates routines to perform preliminary analysis of data corpus to find correlations between input variables and outputs (heat map) as well as to find optimal number of SVD dimensions (plot number of SVD dimensions against prediction accuracies of regression models per dependent variable)
* [svd_varimax.R](src/svd_varimax.R) - performs input features' dimensionality reduction using SVD with subsequent varimax rotation in order to simplify SVD dimensions
* [users_likes_data_set.R](src/users_likes_data_set.R) - holds data set definition with functions to get batches of train/validation samples
* [utils.R](src/utils.R) - provides common utilitites
* [regression_analysis.R](src/regression_analysis.R) - encapsulate experiment with linear/logistic regression predictive models 
* [nn_analysis.R](src/nn_analysis.R) - encapsulate experiment with predictive models based on artificial neural networks
* [mlp.R](src/mlp.R) - encapsulates shallow neural network graph creation
* [dnn.R](src/dnn.R), [3dnn.R](src/3dnn.R) - encapsulates deep neural networks (DNN) graph creation with two and three hidden layers accordingly

Additionality shell scripts provided in order to help with R scripts execution:

* [eval_mlp_1.sh](./eval_mlp_1.sh) - to evaluate shallow neural network
* [eval_dnn.sh](./eval_dnn.sh) - to evaluate DNN with two hidden layers 
* [eval_3dnn.sh](./eval_3dnn.sh) - to evaluate DNN with three hidden layers 


## Running experiments

The detailed instructions how to run experiments present in our research paper. Here we  depict only major steps to be done:

1. The trimmed sparse matrix with users-likes relations must be created using [preprocessing.R](src/preprocessing.R) script.
2. The optimal number of SVD dimensions to be applied to the created users-likes matrix should be found by executing [analysis.R](src/analysis.R) script.
3. With found optimal number of SVD dimensions the dimensionality reduction should be performed using [svd_varimax.R](src/svd_varimax.R) script.
4. The linear/logistic regression analysis can be performed with [regression_analysis.R](src/regression_analysis.R) script using as input the users-likes matrix with reduced features dimensions prepared in previous step.
5. The experiments with predictive models based on neural networks can be executed by running corresponding shell scripts mentioned above.

## Authors

This source code maintained and managed by Iaroslav Omelianenko (NewGround LLC)

[1]:https://www.tensorflow.org/install/
[2]:https://cran.r-project.org/manuals.html
[3]:https://rstudio.github.io/tensorflow/
[4]:http://dataminingtutorial.com
[5]:http://arxiv.org
