# Abstract
It will be shown how to apply basic machine learning algorithms and methodologies to perform analysis of data corpus comprising of collected Facebook likes and psycho-demographic traits of persons associated. Finally, we will analyse performance of classic linear/logistic regression methods applied against more advanced algorithms based on deep machine learning methodologies. The work is also accompanied by corresponding source code in R programming language (R Core Team, 2015) for further analysis by interested parties.

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

Descriptive statistics | Raw matrix | Trimmed Matrix
---------------------- | ---------- | --------------
# of users | 110 728 | 19 742
# of unique Likes | 1 580 284 | 8 523
# of User-Like pairs | 10 612 326 | 3 817 840
Matrix density | 0.006 % | 2.269 %
**Likes per User** | |
	Mean | 96 | 193
	Median | 22 | 106
	Minimum | 1 | 50
	Maximum | 7 973 | 2 487
**Users per Like** | |
	Mean | 7 | 448
	Median | 1 | 290
	Minimum | 1 | 150
	Maximum | 19 998 | 8 445

*Table 1. The descriptive statistics of raw users-likes matrix and trimmed users-likes matrix with minimum users per like threshold set to 150 and minimum likes per user - 50*

### Dimensionality reduction with SVD

The users-likes matrix after preprocessing still have extreme count of features per data sample. In order to make it more maintenable we will consider applying singular value decomposition (SVD, Golub, G. H., & Reinsch, 1970), representing eigendecomposition-based methods, projecting a set of data points into a set of dimensions.
Reducing the dimensionality of data corpus has number of advantages:

1. With reduced features space we can use fewer number of data samples as it is required by most of analysis algorithms that number of data samples exceeds number of features (input variables)
2. It will reduce risk of overfitting and increase statistical power of results
3. It will remove multicollinearity and redundancy in data corpus by grouping related features (variables) in single dimension
4. It will significantly reduce required computational power and memory requirements
5. And finally it makes it easier to analyze data by hand over small set of dimensions as opposite to hundreds or thousands of separate features

### Factor rotation analysis

The factor rotation analysis techniques can be used to simplify SVD dimensions and increase their interpretability by mapping the original multidimensional space into a new, rotated space. Rotation approaches can be orthogonal (i.e., producing uncorrelated dimensions) or oblique (i.e., allowing for correlations between rotated dimensions).

We will apply one of the most popular orthogonal rotation - varimax. It minimizes both the number of dimensions related to each variable and the number of variables related to each dimension, thus improving the interpretability of the data.

For more details on rotation techniques, see (Abdi, 2003).

## Regression analysis

In this section we will consider building of prediction model based on pre-processed data corpus. There is an abundance of methods developed to build prediction models based on large data sets. It's ranging from sophisticated methods such as Deep Learning (Goodfellow-et-al:2016dg), probabilistic graphical models (Daphne-Koller:2012dg), or support vector machines (Cortes-Vapnik:1995dg), to much simpler, such as linear and logistic regressions (Yan-Su:2009dg). 

Starting with simple methods is commonly recommended practice allowing creation of good baseline prediction model with minimal computational efforts. The results obtained from these models can be used later to debug and estimate quality of results obtained by advanced models.

### Cross-Validation

In statistics and machine learning, one of the most common tasks is to fit a "model" to a set of training data, so as to be able to make reliable predictions on general untrained data. In overfitting, a statistical model describes random error or noise instead of the underlying relationship. Overfitting occurs when a model is excessively complex, such as having too many parameters relative to the number of observations. A model that has been overfit has poor predictive performance, as it overreacts to minor fluctuations in the training data.

In order to avoid overfitting, it is necessary to use additional techniques (e.g. cross-validation, regularization, early stopping, pruning, Bayesian priors on parameters or model comparison), that can indicate when further training is not resulting in better generalization.

In this work we will apply k-fold cross-validation to avoid model overfitting. In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining `k-1` subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling (see below) is that all observations are used for both training and validation, and each observation is used for validation exactly once. 10-fold cross-validation is commonly used, but in general `k` remains an unfixed parameter.(Kohavi:1995dg)

### Dimensionality Reduction

In this work we will apply singular value decomposition (SVD) with subsequent varimax factor rotation in order to reduce number of features (variables) in data corpus. The number of the varimax-rotated singular value decomposition dimensions (`K`) has impact on accuracy of model's predictions. In order to find optimal number of SVD dimensions we will perform analysis of relationships between `K` and accuracy of model predictions.

One of the popular methods to find optimal number of SVD dimensions (`K`) is to produce number of models for different values of `K` and plot it against prediction accuracy. Typically prediction accuracy grows rapidly with lower ranges of `K`, and may start decreasing once the number of clusters becomes very large. Selecting a `K` that marks the end of a rapid growth of prediction accuracy values usually offers decent interpretability of the topics. Larger `K` values usually offer better predictive power.(Zhang-Marron-Shen-Zhu:2007dg)

![Prediction accuracies for different values of K SVD dimensions](https://github.com/yaricom/psistats/blob/master/contents/regression/150_50/svd_traits_regression_correlations.png)

*Figure 1. Relationship between the accuracy of predicting psycho-demographic traits and the number of the varimax-rotated singular value decomposition dimensions used. The results suggest that employing `K = 50` SVD dimensions might be a good choice for building models predicting almost all individual's traits of interest, as it offers accuracy that is close to what seems like the higher asymptote for this data. But for Openness, Extroversion, and Agreeableness traits prediction results can be further improved with higher values of `K` SVD dimensions.*

![The heat map of correlations between varimax-rotated singular value decomposition dimensions and scores of psycho-demographic traits of individuals](https://github.com/yaricom/psistats/blob/master/contents/regression/150_50/svd_correlation_hmap.png)

*Figure 2. The heat map presenting correlations between `K = 50` varimax-rotated singular value decomposition dimensions and psycho-demographic traits of individuals. The heatmap suggest that Age, Gender, and Political views of individual has maximal correlation with maximal number of SVD dimensions. The higher correlation will result in higher prediction power of regression model for highly correlated psycho-demographic traits (which will be show later).*

To start analysis run following command from terminal in the project's root directory:

```
Rscript ./src/analysis.R
```

The resulting plots will be saved "Rplots.pdf" file in the project root. This file will include two plots:

1. The plot with relationships between the accuracy of predicting psycho-demographic traits of individuals and the number of the varimax-rotated SVD dimensions used (Figure 1). With this plot it easy to find optimal number of `K` SVD dimensions for maximal predicting power of regression model per specific psycho-demographic trait of individual.

2. The heat map of correlations between scores of individuals on varimax-rotated SVD dimensions and psycho-demographic traits (Figure 2). This plot can be used to visually find most correlated traits of individuals, which results in higher predictive power of regression model for those traits.

## Building regression model and prediction results

In our data corpus we have eight scores for psycho-demographic traits of individual to be predicted. Among those scores six have continuous values and two are has categorical values (binominal: 0, 1). In order to build prediction model for traits with continuous values we will apply linear regression and for traits with categorical values - logistic regression.

### Linear regression

The linear regression is an approach for modeling the relationship between a scalar dependent variable `y` and one or more explanatory variables (or independent variables) denoted `X`. The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.(David-Freedman:2009dg)

In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models.(Hilary-Seal:1967) Most commonly, the conditional mean of `y` given the value of `X` is assumed to be an affine function of `X`; less commonly, the median or some other quantile of the conditional distribution of `y` given `X` is expressed as a linear function of `X`. Like all forms of regression analysis, linear regression focuses on the conditional probability distribution of `y` given `X`, rather than on the joint probability distribution of `y` and `X`, which is the domain of multivariate analysis.

### Logistic regression

The logistic regression, or logit regression, or logit model (David-Freedman:2009) is a regression model where the dependent variable (DV) is categorical.

Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Thus, it treats the same set of problems as probit regression using similar techniques, with the latter using a cumulative normal distribution curve instead. Equivalently, in the latent variable interpretations of these two methods, the first assumes a standard logistic distribution of errors and the second a standard normal distribution of errors. (Rodriguez:2007)

In this article we will consider only specialized binary logistic regression because dependent variables found in our data corpus are binominal, i.e. have only two possible types, "0" and "1".

### Running models and prediction results

In this work we will build and train separate models per each psycho-demographic trait of individual. The prediction accuracy of each model will depend on model's regression type:

*  the prediction power of linear regression models will be measured as Pearson product-moment correlation (Gain:1951)
*  the prediction power of logistic regression models will be measured as area under the receiver-operating characteristic curve coefficient (AUC)(Sing-et-al:2005)

Before executing models make sure that data corpus is pre-processed as described in previous section.

When data corpus is ready the following command can be executed in order to start linear/logistic regression models building and its predictive performance evaluation (run command from terminal in the project's root directory):

```
Rscript ./src/regression_analysis.R
```

The results of predictive performance evaluation will be saved into file $"out/pred\_accuracy\_regr.txt"$. The results of regression models predictions for data corpus trimmed to contain 150 users-per-like and 50 likes-per-user varimax-rotated against `K = 50` SVD dimensions presented in Table 2.

Trait | Variable | Pred. accuracy
----- | -------- | --------------
Gender | gender | 93.65%
Age | age | 61.17%
Political view | political | 68.36%
Openness | ope | 44.02%
Conscientiousness | con | 25.72%
Extroversion | ext | 30.26%
Agreeableness | agr | 23.97%
Neuroticism | neu | 29.11%

*Table 2. The linear and logistic regression models predictive accuracy results per depended variable.*

It can be seen that prediction power of simple linear and logistic regression models applied to the data corpus, differs between each dependent variable and for most outputs its accuracies not enough to be applied for real life predictions. As it was predicted by analysis of SVD correlations heatmap (Figure 1) most accurate predictions was made for 'Gender', 'Age', and 'Political view' of examined individuals with 'Openness' trait following next (but with accuracy lower than simple sampling over normal distribution). In general only linear regression model for 'Gender' may be useful for real life predictions.

# References
Michal Kosinski, Yilun Wang, Himabindu Lakkaraju, and Jure Leskovec, © 2016 American Psychological Association. Mining Big Data to Extract Patterns and Predict Real-Life Outcomes. Psychological Methods 2016, Vol. 21, No. 4, 493–506. http://dx.doi.org/10.1037/met0000105

Lambiotte, R., & Kosinski, M. (2014). Tracking the digital footprints of personality. Proceedings of the Institute of Electrical and Electronics Engineers, 102, 1934–1939. http://dx.doi.org/10.1109/JPROC.2014.2359054

Goldberg, L. R., Johnson, J. A., Eber, H. W., Hogan, R., Ashton, M. C., Cloninger, C. R., & Gough, H. G. (2006). The International Personality Item Pool and the future of public-domain personality measures. Journal of Research in Personality, 40, 84–96. http://dx.doi.org/10.1016/j.jrp.2005.08.007

Golub, G. H., & Reinsch, C. (1970). Singular value decomposition and least squares solutions. Numerische Mathematik, 14, 403– 420. http://dx.doi.org/10.1007/BF02163027

Abdi, H. (2003). Factor rotations in factor analyses. In M. Lewis-Beck, A. E. Bryman, & T. F. Liao (Eds.), The SAGE encyclopedia of social science research methods (pp. 792–795). Thousand Oaks, CA: SAGE.

R Core Team, (2015). R: A language and environment for statistical computing. Vienna, Austria: R Foundation for Statistical Computing. Retrieved from http://www.R-project.org/

Google Brain Team, (2015). TensorFlow™ is an open source software library for numerical computation using data flow graphs. retrieved from https://www.tensorflow.org

[1]:http://dataminingtutorial.com
