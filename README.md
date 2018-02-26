# Portfolio Project : Feature Selection + Classification using the Madelon dataset 

## Problem statement
Investigate the capability of machine learning techniques to idenitfy relavant features in datasets with a large number of features (high dimmensionality) that cannot be handled using conventional techniques.

### Goal:
    1. identification of relevant features. 
    2. demonstrate impact on prediction accuracy 

Two synthetically generated datasets will be used to investigate these techniques ...
    * the Madelon Dataset from UCI Machine Learning Repository. This dataset has 500 features and is the canonical dataset for testing feature selection tecniques.
    * a larger synthically generated Madelon dataset with 1000 features and 200,000 observations     


## Overview
Fundamentaly, this is a binary classification challenge. Using the two datasets of differing sizes, a suite of feature selection options will be developed and their effectiveness will be compared using a variety of classification  models. The aim is to  identify the 'best' features and model for each dataset.


### Scope of solution report.
0. EDA observations for each subset 
1. Establish prediction accuracy benchmarks for raw dataset
2. Implement 3 feature selection models
3. Model selection
4. "Best" Model build & conclusions


## Madelon Dataset

### UCI dataset

Synthetic dataset - from UCI Machine Learning Repository. This is the canonical dataset for testing feature selection tecniques.  
[Madelon](https://archive.ics.uci.edu/ml/datasets/Madelon) 

MADELON is an artificial dataset containing data points grouped in 32 clusters placed on the vertices of a five dimensional hypercube and randomly labeled +1 or -1. The five dimensions constitute 5 informative features. 15 linear combinations of those features were added to form a set of 20 (redundant) informative features. Based on those 20 features one must separate the examples into the 2 classes (corresponding to the +-1 labels). We added a number of distractor feature called 'probes' having no predictive power. The order of the features and patterns were randomized. 

Data available:

|MADELON         |Positive ex. | Negative ex. | Total | Check_sum   
|---------------:|------------:|-------------:|------:|-------------:
|Training set    | 1000        | 1000         | 2000  | 488083511.00    
|Validation set  |  300        |  300         |  600  | 146395833.00      
|Test set        |  900        |  900         | 1800  | 439209553.00     
|All             | 2200        | 2200         | 4400  |     

Number of variables/features/attributes: 
Real: 20  -  5 of which are informative, 15 of which are linear combinations & are therefore redundant
Probes: 480 
Total: 500 

As thre are no target labels available for the test data set we will combine the Training and Validation sets to provide 2600 observations. We will then create our on Training / Test split. This process is explained in the benchmarking step.


### Extended Madelon Dataset (PostgresDB)

Extended Madelon dataset of 200,000 records with a 1000 features, plus one target column,  is available from a GA postgres db hosted on an AWS instance 
    - database name = postgres
    - user name = postgtres
    - table name = madelon
    - AWS instance IP =###########
    
Number of variables/features/attributes: 
Real: 20  -  5 of which are informative, 15 of which are linear combinations & are therefore redundant
Probes: 980 
Total: 1000 

As this is a much larger dataset we will randomly extract 3 samples from the database for the modeling process. To test the final model we will go back to the database to select all 200,000 observations for the selected features.

Sample size required for
        * 95% CI  & 1% margin of error
            * 200,000 -> 9,164   =   4.58%
        * 90% CI & 1% margin of error
            * 200,000 -> 6,506    = 3.25%

3 random samples of 4.6% of total dataset will be used.

Sample acquisition is performed in a separate Jupyter notebook.
Notebook : 00-EDA-PostgresDB-dataset-Sample-extraction.ipynb



## 0. Exploratory Data Analysis

### UCI Dataset
Notebook: 00-EDA-UCI-dataset.ipynb

#### Shape & type of data

2600 Obeservations
    
Target (y):
  Binary Target class = 1 ,-1 
  The classes are evenly distributed in the available dataset.
    
Predictors (X): 
  500 predictor features
  all features are of type integer
    
#### Explore Univariate distributions

First we explore the distributions of each features. Lets look at the mean and standard deviation of each feature...

![](images/00-eda-uci-feature_distributions.jpg)
This graph shows us that the mean and std of the majority of feaures are very similar. We have a small number of features with wider distribtions 
(higher std) and higher mean values.
To see this more clearly lets look at the density funtion for these attributes of the features...

![](images/00-eda-uci-feature_kde.jpg)
    
This highlights the small number of features that with distinct distributions. The plot on the right also highlights the amount of skew in included in the features. 

These findings indicate the  need for some feature enginering:- 
  - A transformation process to remove/reduce the skew
  - Standard scaling to normalise the distribition of each of the variables.

  
#### Explore Multivariate distributions - between features

First we explore the realationship between features. It is difficult to visualize the relationship bewteen large numbers of features, to do this we will look first at the correlation between features and then at the prinicpal components of the variance of the population of feaatures.

##### Correlation between features. 
![](images/00-eda-uci-feature_correlation.jpg)
These plots look at the amount of correlation between feature pairs. From the density plot on the left we can see that the majority of feature pairs have very low correlation ( 1 implies high correlation). The plot on the right shows the feature pairs with the highest correlation. 

##### Principle Component Analysis (PCA)
PCA is a dimmension reduction technique but if can also be used to visualize the strongest directions of variance in a multi-dimmensional feature space.

Think of a swarm of bees where every bee is a feature and where their position at a point in space and time is a feature value (think travel path). The bees may be movig in many different directions and the path of an individal bee does not tell us much about the realationship between the bees. If you look at the direction of travel of the whole swarm you can identify the key axes of travel (principal components of variance) along which they are moving  e.g. towards you, veering to the right and moving upwards . PCA seeks to identify these key axes ( dimensions of variance). If then seeks to identify subsets of bees whose travel is driving the direction the swarm. These subsets of bees (principal components) can be used to explain / predict the direcion of the swarm allowing teh other bees (features) to be dropped. The strenght of  the principal components is measured by how much of the varaince in the feature population (swarm) they predict.rm. 

For the raw UCI dataset the strongest 9 principle components are ...
![](images/00-eda-uci-PC.jpg)
We can see from this that there is  a small number of features that have a strong impact on the first 5 prinical components but the large number of 'nosiey' features are clouding the idenitification  of the priniple component features.

This is reflected in the low amount of the feature space variance that the first 5 principal components explain...
![](images/00-eda-uci-PC-var.jpg)
The feature selection process should remove the noisy features and we will revisit this analysis in the Model Selection process.

#### Explore Multivariate distributions - between features and target
Again this is very difficult to visualise with a large number of features. We will use our principle components to see if there is any clusterig of class values is visible...
![](images/00-eda-uci-PC-corr.jpg)
Some clustering => PC1 PC2 correlated to target. However, difficult to see relationship with the high level of noisy features. We will revist after feature selection.

#### Missing Values treament   
The dataset had one column that was empty where the target variables had been removed. This was removed during the data load process.

#### Feature engineering
As identified in the early analysis we do need two preprocessing steps to ensure the data has a distribution that will not distort the modelling results.

* De-skew data using Box-Cox transformation. This uses the following logic to transform the X values...
```
y = (x**lmbda - 1) / lmbda,  for lmbda > 0
    log(x),                  for lmbda = 0
```
Note: THis transformation requires all values are greater than 0. A small constant is added to all values to ensure this is true. 

If lamda = 0 the transformation is applied value by value and can be applied to the whole X dataset before further modeling  (it is not sample dependent). For this data set applying lmda >0 has little impact on skew so it has been left as 0. 
A normal distribtion has skew=0.
The impact on the level of skew can be seen in the following plot ...
![](images/00-eda-uci-feature_skew.jpg)

* Standard scaling to normalise the distribition of each of the variables.  This is performed for each feature individually by subtracting the mean and dividing by the std. The result is that each feature has a mean=0 and std=1. Clearly this is sample dependent and is done during the modeling process where the datasets are split in a process called cross validation. 


### Postgres : Sample1 Dataset
Notebook: 00-EDA-postgres-sample1.ipynb

THe same EDA process was performed for all 3 samples. Similar observations where made for all three samples. A summary of obersvations for Sample1 only are provided in this report.

#### Shape & type of data

9206 Obeservations
    
Target (y):
  Binary Target class = 0 ,1 
  The classes are evenly distributed in the available dataset.
   0 - 4607
   1 - 4599
    
Predictors (X): 
  1000 predictor features
  all features are of type float
    
#### Explore Univariate distributions

First we explore the distributions of each features. Lets look at the mean and standard deviation of each feature...

![](images/00-eda-sample1-feature_distributions.jpg)
Similarly to te UCI dataset this graph shows us that the mean and std of the majority of feaures are very similar. We have a small number of features with wider distribtions higher std) and higher mean values.
To see this more clearly lets look at the density funtion for these attributes of the features...

![](images/00-eda-sample1-feature_kde.jpg)
This highlights the small number of features that with distinct distributions. The plot on the right also highlights the amount of skew in included in the features. 

As with the UCI dataset these findings indicate the  need for some feature enginering:- 
  - A transformation process to remove/reduce the skew
  - Standard scaling to normalise the distribition of each of the variables.

  
#### Explore Multivariate distributions - between features

First we explore the realationship between features. As with the first dataset we will look first at the correlation between features and then at the prinicpal components of the variance of the population of feaatures.

##### Correlation between features. 
![](images/00-eda-sample1-feature_correlation.jpg)
These plots look at the amount of correlation between feature pairs. From the density plot on the right we can see that the majority of feature pairs have very low correlation ( 1 implies high correlation). The plot on the left shows the feature pairs with the highest correlation. 

##### Principle Component Analysis (PCA)
For the raw Sample1 dataset the strongest 9 principle components are ...
![](images/00-eda-sample1-PC.jpg)
Again we can see from this that there is  a small number of features that have a strong impact on the first 5 prinical components but the large number of 'nosiey' features are clouding the idenitification  of the priniple component features.

This is reflected in the low amount of the feature space variance that the first 5 principal components explain...
![](images/00-eda-sample1-PC-var.jpg)
The feature selection process should remove the noisy features and we will revisit this analysis in the Model Selection process.

#### Explore Multivariate distributions - between features and target
Again this is very difficult to visualise with a large number of features. We will use our principle components to see if there is any clusterig of class values is visible...
![](images/00-eda-sample1-PC-corr.jpg)
Some clustering => PC1 PC2 correlated to target. However, difficult to see relationship with the high level of noisy features. We will revist after feature selection.

#### Missing Values treament   
The dataset had no missing values.

#### Feature engineering
As identified in the early analysis we do need two preprocessing steps to ensure the data has a distribution that will not distort the modelling results.

* De-skew data using Box-Cox transformation. This uses the following logic to transform the X values...
```
y = (x**lmbda - 1) / lmbda,  for lmbda > 0
    log(x),                  for lmbda = 0
```
Note: THis transformation requires all values are greater than 0. A small constant is added to all values to ensure this is true. 

If lamda = 0 the transformation is applied value by value and can be applied to the whole X dataset before further modeling  (it is not sample dependent). For this data set applying lmda >0 has little impact on skew so it has been left as 0. 
A normal distribtion has skew=0.
The impact on the level of skew can be seen in the following plot ...
![](images/00-eda-sample1-feature_skew.jpg)

* As for UCI dataset  - Standard scaling to normalise the distribition of each of the variables is done during the modeling process where the datasets are split in a process called cross validation. 
  

## 1. Establish Benchmarks
Notebook: 01-Benchmarking.ipynb

As we do not yet know which type of model will be most effective for modeling  our dataset , we will use 4 different classification models against which to benchmark the effectiveness of the feature selection and hyperparameter tuneing process.

### Models to use :
    - logistic regression
    - decision tree
    - k nearest neighbors
    - support vector classifier (svc)

We will not tune the hyperparameters of the models at this stage, as we want to establish a baseline performance. 
(For the logistic regression and svc models we will set the hyperparameter C to a high value so that regularisation is minimized - we will look at the impact of increasing regularisation during the model selection process.)

### Metric to use :

In general when assessing the effectiveness of a binary classificaion model we are concerned with both :- 

* Sensitivity : measures the proportion of positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition)

* Specificity : measures the proportion of negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).

There are a number of different metrics that can be used depending of the focus of the particular problem statment. For this exercise we will use the a standard metric called 'accuracy' focused on overall prediction accuracy.

Accuracy = porportion of 'True' results out of the total no of cases predicted ( both true positive and true negative)

Accuracy scores fall within the range of 0 to 1, with 1 reflecting high level prediction accuracy.

### Train / Test split
During the benchmarking process we will create two splits of the data :-   
 * Train dataset = 80% - this will be used to identify our 'best' model
 * Test dataset = 20% - this data set will not be used until we test would final model.
 
The purpose of this is to keep a dataset aside so that we can test our final model knowing that it has not been influenced by the test data. This will give a better indication of how the model will perform on unseen data.

### Benchmark Scores

#### UCI dataset
![](images/01-UCI-benchmarks.png)

![](images/01-UCI-benchmarks-barcht.jpg)

#### Postgres Samples
![](images/01-Samples-benchmarks.png)

![](images/01-Samples-benchmarks-barcht.jpg)

The average benchmark test scores are well below 0.7 and tend to be siginifcantly below the training scores, implying overfitting of the data. 

## 2. Feature Selection

There are 3 main approachs to feature selection. We will use an example of each of these to demonstrate the feature selection process.

### Filter approach
Filter models assess the relevance of features by looking only at the intrinsic properties of the data. In most cases a feature relevance score is calculated, and low-scoring features are removed. Afterwards, this subset of features is presented as input to the classification algorithm. 

Filter models can be either 
 - univariate  -  scoring features one at a time against the target
 - multivariate - use a similar approach but incorporate feature inter-dependencies,

#### Univariate filter
Advantages : 
  - Fast
  - Scalable 
  - Independent of classifier
 
Disadvantages : 
  - Ignores feature dependencies
  - Ignores classifier

#### Multivariate filter
Advantages : 
  - Models feature dependencies
  - Independent of classifier
  - Better computational  complexity than wrapper method ( avoids overfitting)
  
Disadvantages : 
  - Slower than univariate methods
  - Less scalable than univariate methods
  - Ignores classifier

Sklearn offers univariate feature filter as a class named SelectKBest which we will use for one of our models. 


### Wrapper approach
Wrapper models setup a search procedure in the space of possible feature subsets. Various subsets of features are generated and evaluated by training and testing a specific classification model. To search the space of all feature subsets, a search algorithm is ‘wrapped’ around the classification model. However, as the space of feature subsets grows exponentially with the number of features, heuristic search methods are used to guide the search for an optimal subset. These search methods can be divided in two classes: deterministic (will always produce the same outcome) and randomized search algorithms. 

#### Deterministic search methods
Advantages : 
  - Simple
  - Interacts with classifier
  - Models feature dependencies
  - Less computationally expensive than randomized models

Disadvantages : 
  - Risk of overfitting
  - More prone than randomized algorithms to getting stuck in local optimum greedy search)
  - Classifier dependent selection

#### Random search methods
Advantages : 
  - Less prone to local optima
  - Interacts with classifier
  - Models feature dependencies
  
Disadvantages : 
  - Computationally expensive
  - Classifier dependent selection
  - Higher risk of overfitting than deterministic mode

Sklearn offers a wrapper model with a deterministic search method as a class named RFE (Recursive Feature Elimination) which we will use for one of our models.

### Embedded type
Embedded models use the  search for an optimal subset of features built into a classifier construction. Just like the wrapper approach, embedded approaches are specific to a given learning algorithm but embedded methods have the advantage that they are far less computationally intensive than wrapper methods. 

Advantages : 
  - Interacts with classifier
  - Models feature dependencies
  - Better computational complexity than wrapper models

Disadvantages : 
  - Classifier dependent selection
  
Sklearn offers an embedded model feature selector as a class named SelectFromModel which we will use for one of our models.


### Implementation

For each of these models we will create a pipeline that will run within a gridsearch with 5 cross validations. 

A gridsearch enables the pipeline to be run multiple times with different combinations of hyperparameters.

Cross validations involve randomly splitting the data set into train and validation sets and running the model on each split of the data. The "train" data is used to 'fit' the model and the 'validation' data is used to test the effectiveness of the model. By running the model on multiple splits of the data we can use the mean scores to ensure the model will generalise to other data.  

Each step in the pipeline ...
 - takes the data output from the previous step
 - applies the model in the step to fit and tranform the data
 - passes the transformed data onto the next step.
...the final step must be an estimator.

The pipeline for each model will contain
  - step1 : apply Standard Scaling
  - step2 : feature selection model
  - step3 : classification model to assess effectiveness of feature selection

We will use each of the the naieve benchmark models to score the features selected by each feature selection model. The results from these models will be assessed in the Model Selection process.

#### SFM -  SelectFromModel
Notebook: 02-Feature-Selection-SelectFromModel.ipynb

Hyperparatmeters:
  - threshold: [.25,.3,.4,.45,.5,.6] ( this was an iterative process to find the threshold that reduced the feature set to the desired size)
  - estimator = a Classifier  - LogisticRegression()
    - Set penalty='l1' (Lasso).
    - Set solver='liblinear' (required for the Lasso penalty).
    - try range of C  in  cnp.logspace(1,3,3)


#### SelectKbest
Notebook: 02-Feature-Selection-SelectKBest.ipynb

Hyperparatmeters: 
   - k threshold : 
    - number of features to select and pass to next method ( based on lowest pvalues). 
    -  We are looking for 5 informative and 15 redundant features so we will try a range of values 
           k = [5,10,15,20]
   - scoring function: 
    - f_classifi - used for classification problems with Categorical target and continous features
    - Function used to calculate pvalues of ANOVA F-test. 
    - Low pvalue implies => reject null hypothesis
           null hypothesis = feature and tartget have not statistically different.

#### RFE (Recursive Feature elimination)
Notebook: 02-Feature-Selection-RFE.ipynb

Hyperparamters:
  - Estimator: LogisticRegression - estimator needs to provide feature importance or coeff
    - penalty=l1 . ( l1 regularisation the allows coeffs to go to zero)
    - C regularisation strength  - try [10,100,1000] , small c= high regularisation
    - n-features_to_select : try [5,10 ]

Range of Parameters was restricted due to long runtimes.

The disadvantages of this model were evident with the larger sized datasets from postgresDB. This model failed to complete for these datasets - further tuning and a different choice of esimator may have helped.



## 3. Model assessment
Notebook: 03-Model-Selection_UCI.ipynb
Notebook: 03-Model-Selection_Postgres.ipynb

### Model assessment process

1. Review the results of the feature selection process and identify the set(s) of 'best' features for the final model for both the UCI and postgres datasets.

2. Rerun PCA model and assess whether to include in model assessment.

3.  Using each of the 'best' feature set(s) run a gridsearch on each of the classification models used in the benchmarking asssessment. 

#### Models: 
    - logistic regression
    - decision tree
    - k nearest neighbors
    - support vector classifier (svc)

Use the gridsearch process to 'tune' 
   - the selecion of the best 'best' feature set
   - classification model paramter settings
   - inclusion or exclusion of PCA as an additional preprocessing step

4. Review all the results from the model assessment process and determine final model specifcations
   - best feature set
   - preprocessing  - Scaling / PCA
   - classification model and parameter settings

### Feature Selecton results

#### UCI Summary of results
![](images/03-Model-Selection-uci-feature-selection-summary.png)
The classification models used to score the feature selection have clearly influenced the 'best' feature selection. 

The following range of features were selected.
![](images/03-Model_Selection_uci_featuresets.png)

Looking accross all sets we can see which features were selected most frequently.
![](images/03-Model_Selection-uci-bf.jpg)
As there is no way to clearly identify the top 5 or top 10 we will take three cutoff points based on the freqency of occurence of a feature in a 'selected' set. Each of these 3 sets will be further assessed in the model assessment.

#### UCI best feature sets
All = [ 28,  48,  64, 105, 128, 153, 241, 281, 318, 323, 336, 338, 378,
            384, 433, 442, 451, 453, 472, 475, 493]

Frequency > 8
top 5 = [48, 105, 128, 338, 378]

Frequency >6
top 10= [28, 48, 105, 128, 241, 336, 338, 378, 433, 451]

#### Postgres Samples - Summary of results
![](images/03-Model-Selection-samples-feature-selection-summary.png)
The classification models used to score the feature selection have clearly influenced the 'best' feature selection. 

The following range of features were selected.
![](images/03-Model_Selection-sample1-bf.jpg)

Looking accross all sets we can see which features were selected most frequently. 
![](images/03-Model_Selection-sample1-featuresets.png)
As there is no way to clearly identify the top 5 or top 10 we will take three cutoff points based on the freqency of occurence of a feature in a 'selected' set. Each of these 3 sets will be further assessed in the model assessment.

#### Postgres Samples best feature sets
All = [257, 269, 315, 336, 341, 395, 494, 504, 526, 646, 667, 681, 701,
            724, 736, 769, 808, 829, 867, 920]

Frequency > 10  ( actually 10 features)
top 5 = [269, 315, 341, 681, 701, 736, 769, 808, 829, 920]

Frequency >8  ( actually 14 features)
top 10= [257, 269, 315, 336, 341, 504, 681, 701, 736, 769, 808, 829, 867,
            920]

### Review PCA 
#### UCI
The PCA model we looked at in the EDA process was rerun with the 'best' feature selection.
![](images/03-Model_Selection-UCI-pc.jpg)
![](images/03-Model_Selection-UCI-pc_var.jpg)
The top 2 principal components now account for nearly all the variance in the feature space. We will include PCA as an addtional dimmension reduction process to assess in the model assessment. 
![](images/03-Model_Selection-UCI-pc_corr.jpg)
There is some evidence of clustering of classes when looking at PC1 and PC2 correlation but this is stll not very strong.

#### Postgres Samples
![](images/03-Model_Selection-sample1-pc.jpg)
![](images/03-Model_Selection-sample1-pc_var.jpg)
The top 5 principal components now account for nearly all the variance in the feature space. We will include PCA as an addtional dimmension reduction process to assess in the model assessment. 
![](images/03-Model_Selection-sample1-pc_corr.jpg)
There is some evidence of clustering of classes when looking at PC1 and PC2 correlation but this is stll not very strong.

#### Model hyperparamter tuning
### UCI
The boxplot below showsa summary of the results of the model assessment and tuning.
![](images/03-Model_Selection-UCI-modelscores_boxplot.jpg)

Model with best mean test score :
  - Prepocessing = Standard scaling , do not include PCA
  - Best feature set = 21 features
  [ 28,  48,  64, 105, 128, 153, 241, 281, 318, 323, 336,338, 378, 384, 433, 442, 451, 453, 472, 475, 493]
  - Best model = KNeighborsClassifer
  - Best model parameters = n-neighbors=7

#### Postgres Samples
The boxplot below shows a summary of the results of the model assessment and tuning.
![](images/03-Model_Selection-samples-modelscores_boxplot.jpg)

Model with best mean test score :
  - Prepocessing = Standard scaling , do not include PCA
  - Best feature set =  10 features
  [269, 315, 341, 681, 701, 736, 769, 808, 829, 920]
  - Best model = KNeighborsClassifer
  - Best model parameters = n-neighbors=9

## 4. Final Model Build
Notebook: 04-Build Model-UCI.ipynb
Notebook: 04-Build Model-Postgres_data.ipynb

Final model build based on specifications defined in step 3. Model assessment.

#### Process:
- Pipeline built for specified steps
- 5 fold Cross fold vaidation run on training data to fit 'final' model.
- Final model assessed by scoring Test data set aside in Benchmarking step .

#### Results
Final model accuracy scores.

![](images/04-Final-Model-Results.png)

Cross validaion scores are average scores for the 5 train/test data splts. 

#### Conclusions

The feature selection process successfully identifed the 20 real features from the 500 UCI features and the 1000 Postgres features, resulting in siginigicant improvements in model accuracy. 

THe features selection process was not able to conclusively separate the 5 informative features from the 15 redundant features although some narrowing of the 20 features was achieved.

Further tunning of the n-neighbors parameter achieved addtional improvements in model accuracy. 
