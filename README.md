# Overview
-------------
In this work I'll try to accomplish performance analysis of various machine learning (ML) algorithms and methodologies over data corpus comprising of Facebook likes and psychodemographic traits of users associated with likes. The psychodemographic traits based on the five-factor (i.e., Openness, Conscientiousness, Extroversion, Agreeable- ness, and Neuroticism) model of personality (Goldberg et al., 2006) and was collected through Facebook application (Kosinski et al., 2016). The main goal of this research is to compare various ML algorithms against data corpus and to find best solution.
# Data Set
-------------
The sample data set using in this research may be acquired from [TUTORIAL: Mining Big Data to Extract Patterns and Predict Real-Life Outcomes][1], which is kindly provided by M. Kosinski.
The following three files can be downloaded from the mentioned website:
1. _users.csv:_ contains psychodemographic user profiles. It has nu = 110,728 rows (excluding the row holding column names) and nine columns: anonymized user ID, gender (“0” for male and “1” for female), age, political views (“0” for Democrat and “1” for Republican), and scores of five-factor model of personality (Goldberg et al., 2006).
2. _likes.csv:_ contains anonymized IDs and names of nL = 1,580,284 Facebook Likes. It has two columns: ID and name.
3. _users-likes.csv:_ contains the associations between users and their Likes, stored as user–Like pairs. It has nu-L = 10,612,326 rows and two columns: user ID and Like ID. An existence of a user–Like pair implies that a given user had the corresponding Like on their profile.

# Data preprocessing
-------------
In order to use provided data corpus it should be preprocessed with following steps:
1. Construction of sparse users-likes matrix which presents many-to-many relationships between users and their digital footprints in the form of collected Facebook likes. The constructed matrix is extremely big with high sparsity, so it is appropriate to operate with it and store it in _sparse data format_, which is optimized for such kind of data.
2. Trimming of sparse users-likes matrix in order to exclude rare data which has no significance
3. Dimensionality reduction in order to reduce extremely large number of features in the generated data corpus
4. Factor rotation analysis to simplify SVD dimensions

### Construction of sparse users-likes matrix and its trimming
The matrix can be constructed by accompanying script writen in R language [preprocessing.R](https://github.com/yaricom/psistats/blob/master/src/preprocessing.R). In order to use this script please make sure that **input_data_dir** variable in the [config.R](https://github.com/yaricom/psistats/blob/master/src/config.R) points to the root directory where sample data corpus in form of .CSV files unpacked.
To start preprocessing and trimming issue run following command from terminal starting from the project root directory:
```
$Rscript ./src/preprocessing.R -u 150 -l 50
```
where: **-u** is the minimum number of users per like, and **-l** is the minimum number of likes per user to keep in resulting matrix
As result of users-likes matrix trimming we obtain significantly reduced data corpus which has lower demands for computational resources and mush more useful for manual analysis to extract specific patterns. The descrptive statistics of users-likes matrix before and after trimming present in table 1.
| Descriptive statistics | Raw Matrix |Trimmed Matrix|
|------------------------|:----------:|:------------:|
|# of users|110 728|19 742|
|# of unique Likes|1 580 284|8 523|
|# of User-Like pairs|10 612 326|3 817 840|
|Matrix density|0.006 %|2.269 %|
|**Likes per User**|
|Mean|96|193|
|Median|22|106|
|Minimum|1|50|
|Maximum|7 973|2 487|
|**Users per Like**|
|Mean|7|448|
|Median|1|290|
|Minimum|1|150|
|Maximum|19 998|8 445|
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
The factor fotation analysis techniques can be used to simplify SVD dimensions and increase their interpretability by mapping the original multidimensional space into a new, rotated space. Rotation approaches can be orthogonal (i.e., producing uncorrelated dimensions) or oblique (i.e., allowing for correlations between rotated dimensions).
We will apply one of the most popular orthogonal rotation - varimax. It minimizes both the number of dimensions related to each variable and the number of variables related to each dimension, thus improving the interpretability of the data.

# References
-------------
Michal Kosinski, Yilun Wang, Himabindu Lakkaraju, and Jure Leskovec, © 2016 American Psychological Association. Mining Big Data to Extract Patterns and Predict Real-Life Outcomes. Psychological Methods 2016, Vol. 21, No. 4, 493–506. http://dx.doi.org/10.1037/met0000105

Goldberg, L. R., Johnson, J. A., Eber, H. W., Hogan, R., Ashton, M. C., Cloninger, C. R., & Gough, H. G. (2006). The International Personality Item Pool and the future of public-domain personality measures. Journal of Research in Personality, 40, 84–96. http://dx.doi.org/10.1016/j.jrp.2005.08.007

Golub, G. H., & Reinsch, C. (1970). Singular value decomposition and least squares solutions. Numerische Mathematik, 14, 403– 420. http://dx.doi.org/10.1007/BF02163027

R Core Team. (2015). R: A language and environment for statistical computing. Vienna, Austria: R Foundation for Statistical Computing. Retrieved from http://www.R-project.org/

[1]:http://mypersonality.org/wiki/doku.php?id=mining
