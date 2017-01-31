# Abstract
It will be shown how to apply basic machine learning algorithms and methodologies to perform analysis of data corpus comprising of collected Facebook likes and psycho-demographic traits of persons associated. Finally, we will be analyse performance of classic linear/logistic regression methods applied against more advanced algorithms based on deep machine learning methodologies. The work is also accompanied by corresponding source code in R programming language (R Core Team, 2015) for further analysis by interested parties.

# Overview
Recent research demonstrates applicability of machine learning for analysis of user generated content in order to study important personality traits (Lambiotte & Kosinski, 2014). It was demonstrated that strong statistical correlation exists between digital footprints of individuals and their psychometric traits based on the five-factor (i.e., Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism) model of personality (Goldberg et al., 2006).

In this work we will concentrate our efforts on finding statistical correlations between digital footprints of individual in form of Facebook likes and her psychometric traits. The data corpus with collected Facebook likes and psychometric scores of individuals kindly made publicly available by M. Kosinski and may be requested from corresponding web site: [http://dataminingtutorial.com][1].

This article will show how to use classical machine learning analysis based on linear/logistic regression approach proposed by M. Kosinski (Kosinski et al., 2016) and extends it further in order to explore how analysis performance will change by applying advanced deep machine learning algorithms.

We will start with data corpus preprocessing and constructing users-likes sparse matrix and proceed further with applying various machine learning analysis methodologies to resulting data corpus.

The provided source code is written in R programming language which is highly optimized for statistical data processing and allows to apply advanced deep machine learning algorithms by providing bridge to TensorFlow (Google Brain Team, 2015) framework.


# Data Corpus Preparation
In this section we will create input data corpus from downloadable data set and preprocess it in order to simplify further analysis by machine learning algorithms. 

## Data Set Description 
The data set kindly provided by M. Kosinski and used in this article contains psycho-demographic profiles of nu = 110 728 Facebook users and their Facebook likes. For simplicity and manageability, the sample is limited to U.S. users. The following three files can be downloaded:

1. _users.csv:_ contains psycho-demographic user profiles. It has nu = 110 728 rows (excluding the row holding column names) and nine columns: anonymized user ID, gender (“0” for male and “1” for female), age, political views (“0” for Democrat and “1” for Republican), and scores of five-factor model of personality (Goldberg et al., 2006).
2. _likes.csv:_ contains anonymized IDs and names of nL = 1 580 284 Facebook Likes. It has two columns: ID and name.
3. _users-likes.csv:_ contains the associations between users and their Likes, stored as user–Like pairs. It has nu-L = 10 612 326 rows and two columns: user ID and Like ID. An existence of a user–Like pair implies that a given user had the corresponding Like on their profile.

## Data preprocessing
In order to use provided data corpus it should be preprocessed with following steps:

1. Construction of sparse users-likes matrix which presents many-to-many relationships between users and their digital footprints in the form of collected Facebook likes. The constructed matrix is extremely big with high sparsity, so it is appropriate to operate with it and store it in _sparse data format_, which is optimized for such kind of data.
2. Trimming of sparse users-likes matrix in order to exclude rare data which has no significance
3. Dimensionality reduction in order to reduce extremely large number of features in the generated data corpus
4. Factor rotation analysis to simplify SVD dimensions

### Construction of sparse users-likes matrix and its trimming
The matrix can be constructed by accompanying script written in R language [preprocessing.R](https://github.com/yaricom/psistats/blob/master/src/preprocessing.R). In order to use this script please make sure that **input_data_dir** variable in the [config.R](https://github.com/yaricom/psistats/blob/master/src/config.R) points to the root directory where sample data corpus in form of .CSV files unpacked.
To start preprocessing and trimming run the following command from terminal in the project's root directory:
```
$Rscript ./src/preprocessing.R -u 150 -l 50
```
where: **-u** is the minimum number of users per like, and **-l** is the minimum number of likes per user to keep in resulting matrix
As result of users-likes matrix trimming we obtain significantly reduced data corpus which has lower demands for computational resources and mush more useful for manual analysis to extract specific patterns. The descriptive statistics of users-likes matrix before and after trimming present in Table 1.
```
Descriptive statistics  Raw matrix  Trimmed Matrix
# of users		        110 728         19 742
# of unique Likes	   1 580 284         8 523
# of User-Like pairs  10 612 326      3 817 840
Matrix density		    0.006 %        2.269 %
Likes per User
	Mean		           96            193
	Median		           22            106
	Minimum		            1             50
	Maximum		         7 973          2 487
Users per Like
	Mean		            7            448
	Median		            1            290
	Minimum		            1            150
	Maximum		        19 998          8 445
```
*Table 1. The descriptive statistics of raw users-likes matrix and trimmed users-likes matrix with minimum users per like threshold set to 150 and minimum likes per user - 50*

### Dimensionality reduction with SVD
The users-likes matrix after preprocessing still have extreme count of features per data sample. In order to make it more maintenable we will consider applying singular value decomposition (SVD, Golub, G. H., & Reinsch, 1970), representing eigendecomposition-based methods, projecting a set of data points into a set of dimensions.
Reducing the dimensionality of data corpus has number of advantages:

1. With reduced features space we can use fewer number of data samples as it is required by most of analysis algorithms that number of data samples exceeds number of features (input variables)
2. It will reduce risk of overfitting and increase statistical power of results
3. It will remove multicollinearity and redundancy in data corpus by grouping related features (variables) in single dimension
4. It will significantly reduce required computational power and memory requirements
5. And finaly it makes it easier to analyse data by hand over small set of dimensions as oposite to hundreds or thoushands of separate features

### Factor rotation analysis
The factor rotation analysis techniques can be used to simplify SVD dimensions and increase their interpretability by mapping the original multidimensional space into a new, rotated space. Rotation approaches can be orthogonal (i.e., producing uncorrelated dimensions) or oblique (i.e., allowing for correlations between rotated dimensions).

We will apply one of the most popular orthogonal rotation - varimax. It minimizes both the number of dimensions related to each variable and the number of variables related to each dimension, thus improving the interpretability of the data.

For more details on rotation techniques, see (Abdi, 2003).

# References
Michal Kosinski, Yilun Wang, Himabindu Lakkaraju, and Jure Leskovec, © 2016 American Psychological Association. Mining Big Data to Extract Patterns and Predict Real-Life Outcomes. Psychological Methods 2016, Vol. 21, No. 4, 493–506. http://dx.doi.org/10.1037/met0000105

Lambiotte, R., & Kosinski, M. (2014). Tracking the digital footprints of personality. Proceedings of the Institute of Electrical and Electronics Engineers, 102, 1934–1939. http://dx.doi.org/10.1109/JPROC.2014.2359054

Goldberg, L. R., Johnson, J. A., Eber, H. W., Hogan, R., Ashton, M. C., Cloninger, C. R., & Gough, H. G. (2006). The International Personality Item Pool and the future of public-domain personality measures. Journal of Research in Personality, 40, 84–96. http://dx.doi.org/10.1016/j.jrp.2005.08.007

Golub, G. H., & Reinsch, C. (1970). Singular value decomposition and least squares solutions. Numerische Mathematik, 14, 403– 420. http://dx.doi.org/10.1007/BF02163027

Abdi, H. (2003). Factor rotations in factor analyses. In M. Lewis-Beck, A. E. Bryman, & T. F. Liao (Eds.), The SAGE encyclopedia of social science research methods (pp. 792–795). Thousand Oaks, CA: SAGE.

R Core Team, (2015). R: A language and environment for statistical computing. Vienna, Austria: R Foundation for Statistical Computing. Retrieved from http://www.R-project.org/

Google Brain Team, (2015). TensorFlow™ is an open source software library for numerical computation using data flow graphs. retrieved from https://www.tensorflow.org

[1]:http://dataminingtutorial.com
