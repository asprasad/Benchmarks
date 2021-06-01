# Kaggle Benchmark Descriptions
Descriptions of the benchmarks from Kaggle and some analysis of each of them.

## Instant Gratification
### Functionality
* **Read Data:** Simply reads the data in from the CSV files train and test. No other processing is performed.
* **Predictive Models:**
  * Uses pseudo labeling -- a process where a model is first built and its "confident" predictions are used to generate more training data from test data.
    * Here, any test data predicted to belong to a certain class with probability >= 0.99 is added to the training data.    
  * The first observation is that the column 'wheezy-copper-turtle-magic' takes an integer value between 1 and 512.
  * The algorithm first builds 512 different models based on the value of this column. All the models are [**QuadraticDiscriminantAnalysis**](https://en.wikipedia.org/wiki/Quadratic_classifier). It uses the sklearn function [**QuadraticDiscriminantAnalysis**](https://scikit-learn.org/stable/modules/lda_qda.html#lda-qda).
    * it filters the training and test data based on the value of 'wheezy-copper-turtle-magic' and builds one model for each of these.
   * The algorithm then builds and trains 512 models again based on the augmented training data (labelled with the predictions generated in the previous step).
   * The predictions of these 512 models are combined to generate the final predictions. 
### Profiling ([Visualization](https://github.com/asprasad/Benchmarks/blob/main/kaggle/instant-gratification/cdeotte/pseudo-labeling-qda-0-969-refactored.py.svg))
* **Reading Data :** Takes about 10% of the total time. All the time is spend reading from disk.
* **Feature Selection :** The feature selection (fit + transform) using VarianceThreshold (sklearn) takes about 17% of the time (fit takes 7% and transform 10%). 
* **Training :** A total of about 36% of the time is spent on training (combined training time for both stages). About 30% (>80% of the training time) is spend on SVD which is the solver used for QDA.
* **Filtering And Other Pandas Stuff:** Takes about 23% of the time. There are two calls to two different __getitem__ functions -- pandas/core/frame.py and pandas/core/indexing.py.
* **Inference :** Uses predict_proba on the QDA object. This takes a total of ~5% of the time.
* **Misc :** There seems to be a total of about 2-3% time spent on other pandas calls in the final model building. I'm not sure why this doesn't show up in the first round of model building.
* [Opt] All the models that are being built are independent. Training the models and inference can be completely parallelized.
* [Opt] Each iteration is filtering the data based on a certain feature value. This is creating a copy of the data and then feature selection is performed on it. We could potentially merge feature selection (the variance thresholding) for all 512 values of the "filtering feature" so that we only walk the data set once to compute the aggregates required to select features. 

## PUBG
### Functionality
* **Read Data:** Just reads the data from the test and train CSV files. Performs some basic filtering. Interestingly, there is a memory reduction function that casts various columns to specific data types. Apparently, the process uses too much memory without this. (TODO is this potentially an opportunity for optimizations like tiling?)
* **Feature Augmentation:** After reading the CSVs, the algorithm constructs a set of additional features by combining columns in various ways. It roughly has two parts
  * First, a new set of features is constructed by arithmetically combining columns from the input data.
  * Input rows are grouped based on some keys and each group is reduced in different ways (min, max, mean, std etc.). Then the aggregates of each group are appended to each row of the group (using a join).
  * [Opt] For each different aggregation, the grouping is being re-run. Also, the join is a major bottleneck. If we were to join the aggregate tables first and then perform a single join with the input table, that would be much faster than repeatedly joining the aggregates with the input data.
* **Predictive Models:** There is a single predictive model -- a gradient boosted tree ensemble. The single model is trained with all the augmented training data and then used to predict the test inputs.
### Profiling ([Visualization](https://github.com/asprasad/Benchmarks/blob/main/kaggle/pubg/kamalchhirang/5th_place_solution_0_0184_score.py.svg))
* **Read Data:** The read + feature generation takes about 9% of the total time. This time is dominated by the joins (~7%).
  * See above for some possible optimizations.
* **Training:** The training is the biggest bottleneck. Takes about 80% of the time. Almost all of this is spent in the lightgbm.train (~78%).
  * LightGBM.train seems to have an input called "num_threads". Is this some kind of parallelization?
*  **Inference:** Takes about 10.5% of the total time of which about 7.5% is the actual inference (model.predict) and 3% is reading the test data and adding the derived features.

## Santander
### Functionality
* **Read Data and Feature Augmentation:** 
  * Read the CSV into a dataframe. Also does some basic filtering based on whether some feature values are unique (not sure how this works since the results seem to be discarded). 
  * There are 200 input features. 200 new features, one for each input feature, are computed based on the histogram of the corresponding input feature. (These are the magic features).
* **Predictive Models:** The application compares two prediction mechanisms -- 
  * In the first method, the magic features are used. 200 different lightgbm models, one for each input feature magic feature pair, to predict the target value are constructed. The predictions of these models are then combined to get a final prediction. They are combined using logistic regression. 
  * The second method is the same as above except that it only uses the input features and not the magic features.
 Additionally, for each variable (or variable pair), 5 folds of the training data are constructed. A lightgbm model is trained for each fold and the prediction is computed as the average of these models.
### Profiling ([Visualization](https://github.com/asprasad/Benchmarks/blob/main/kaggle/santander-transaction-pred/cdeotte/200-magical-models-santander-0-920-refactored.py.svg))
I only profiled the application for model construction and inference for the first 5 features. Therefore, the data read and feature augmentation times should be even more insignificant than what is listed below (it says a total of ~9% below, but I would expect something like 0.2% when the whole application is run, i.e. the whole of the 200 iterations).
* **Read Data and Feature Augmentation:** 
  * Reading and filtering take about 5% of the total time. Reading from the disk is about 3.5% and the rest is almost completely the unique value computation.
  * Feature augmentation is about 3.5% of the total time.
* **Training:** Training is again the biggest bottleneck taking about 80% of the time. Even though the model used is the same as the one in PUBG, the split of time within the train method is very different. This is possibly because we are training several small models here as opposed to a single large model in PUBG. 
  * There is a non trivial cost of filtering out only the required columns from the input(or augmented) dataframe. This takes a total of about 4% of the time.
  * [Opt] All the models that are being built are independent. Training the models and inference can be completely parallelized.
* **Inference:** Inference takes only ~5% of the time. The logistic regression used to combine the predictions of the various models takes negligible time.

## March Madness
The aim of this program is to construct a model to predict the competitiveness of a basketball match. The following is a summary of what this benchmark does
* A "competitiveness" label is decided for several past games based on known game characteristics (like lead changes, points difference etc).
* A large set of features are computed on the existing games.
* A gradient boosting based model is trained for the data to predict whether or not a game will be competitive and 5-fold cross validation is performed.
### Functionality
* **Read Data and Feature Engineering :** 
  * Reads data from one file at a time and computes several additional features using pandas APIs.
  * For example, it computes the following
   * The score at each "event" in the game (make_scores -- dominated by a groupby).
   * Computes the number of times the lead changes in each focus period (lead_changes -- again dominated by groupby).
   * The count of events of interest in each period (event_count)
   * For each game, compute statistics over the previous 30 days for both teams (prepare_competitive, rolling_stats -- dominated by reductions over the windows).
 * **Training and Cross Validation :** The model used is gradient boosting (XGBoost). A single model is trained with combined data from all tournaments. Cross validation is used to compute the accuracy. There is no real inference as such.
### Profiling ([Visualization](https://github.com/asprasad/Benchmarks/blob/main/kaggle/march-madness/lucabasa/quantify-the-madness-a-study-of-competitiveness.py.svg))
The profiles were collected while running the benchmark only on data from 2020. The actual dataset has data from 2015-2020. But I expect that adding more data will scale all parts of this application uniformly.
* **Read Data and Feature Engineering :** This takes about 60% of the total time.
  *  prepare_competitive (computation of the rolling stats), lead_changes, make_scores, event_count account for almost all the time spent reading data and constructing features. Bottlenecks of each are listed above.
  *  [Opt] The feature construction for each input csv (there are 100s) is done separately and then all the tables are concatenated. Each can be processed in parallel. Additionally, pipeline parallelism for computation of various features is also possible between different input files. Data-parallelism also exists within the operators.
  *  [Opt] Specifically for the reduction over 30 day intervals, these reductions can be parallelized over different windows and the different aggregations combined (rather than going over the data multiple times).
  *  [Opt] It is probably also possible to perform some query rewrite optimization since most of the time is spend in pandas.
*  **Training and Cross Validation :**
  *  Training takes about 23% of the total time.
  *  Computing partial dependence plots takes ~17% (this is probably not needed and is part of model evaluation).

## House Prices
### Functionality
This application compares the performance of different models in terms of predicting house prices using various features. Cross validation is used to predict performance.
* **Read Data and Feature Engineering :**
  * Reads data from a csv file
  * Fills in missing values using different techniques like median and mean. 
  * Transforms some columns using functions like log.
  * Overall, a negligible amount of time is spent reading data and manipulating it.
* **Training and Cross Validation :**
  * Several models are compared in terms of their prediction accuracy. The following are the standalone models that are evaluated.
   * Lasso
   * Elastic net
   * Kernel ridge regression
   * Sklearn gradient boosting (GradientBoostingRegressor)
   * XGBoost
   * LightGBM
   * Averaging model : Averages the prediction of Elastic net, sklearn gradient boost, kernel ridge regression and lasso
   * A stacked model : The stacked model uses Elastic net, sklearn gradient boost, kernel ridge regression as the first level predictors and then combines their predictions using a lasso predictor (as opposed to just averaging which was done in the Averaging model)
   * All these models are evaluated using the sklearn function cross_val_score.
     * TODO This seems to do some fancy parallelization under the hood. Need to figure out what its doing. (Uses joblib)
  * Finally, a stacked ensemble model is trained and evaluated (stacked model + XGBoost + lightGBM). The stack model uses Elastic net, sklearn gradient boost, kernel ridge regression as the first level predictors and then combines their predictions using a lasso predictor (as opposed to just averaging which was done in the Averaging model). Additionally, the prediction of the stacked model is linearly combined (using some constant coefficients) with predictions from XGBoost and LightGBM.
    * This model is only evaluated using a root mean square error on the training data. No cross validation is performed.
### Profiling
* Reading data and manipulating features takes negligible time (I can't even find it on the flame graph). 
* Evaluating the stacked model takes the longest time (about 60% of the time), followed by evaluating the averaged model (~12%), the sklearn gradient boosting model (9.5%), the XGBoost model (4.5%) and LightGBM (0.5%).
  * Presumably, these times are dominated by training times. But this is not clear from the profile since the callstacks are hidden by the parallel execution inside cross_val_score.
* Evaluating the stacked ensemble model takes about 13.2% of the time of which a majority is taken by the training (fit). How much exactly is not clear again because of some weirdness in the callstacks.
  * This takes less time than evaluating the stacked model because here we don't do 5-fold cross validation. All three models (stacked, XGBoost and LightGBM) are trained on the full training set and the RMSE is evaluated.
* [Opt] Can some cross validation specific optimizations be done for training? Are there things that can be reused across folds?
* [Opt] Will training all these models together on a specific data set be faster than training them one after the other?  

## Titanic
### Functionality
This application compares various models for the titanic dataset and finally builds a voting classifier to make predictions. The aim is to predict whether a given person survived the titanic disaster given various features describing the person (age, sex, ticket class etc.).
* **Read Data and Feature Engineering: **
  * Reads data from training and test CSV files into pandas dataframes. 
  * Fills in missing values for various features.
  * Transforms some features and generates a few new ones (like family size).
  * Does one hot encoding on some categorical features.
  * Again, the time spent in this part of the application is negligible.
* **Modeling and Cross Validation: **
  * The application evaluates several models on the titanic dataset. It uses cross_val_score and 10-fold cross validation to evaluate these models. 
    * SVC
    * Decision Tree
    * AdaBoost 
    * Random Forest
    * Extra Trees
    * Gradient Boosting
    * Multiple layer perceprton (neural network)
    * KNN
    * Logistic regression
    * Linear Discriminant Analysis
   * The best models from these are picked and a hyperparameter search is performed on them. This uses GridSearchCV and 10-fold CV.
     * Again, there is some parallel execution within the grid search and this confuses the profiler. Its not obvious what the bottlenecks within these calls are.
     * The best models are AdaBoost, RandomForest, ExtraTrees and GradientBoosting
   * The configurations of the 4 best models above are combined in a voting classifier. This is again trained on the whole training dataset.
 
