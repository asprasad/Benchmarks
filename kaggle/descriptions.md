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
### Profiling
* **Reading Data :** Takes about 10% of the total time. All the time is spend reading from disk.
* **Feature Selection :** The feature selection (fit + transform) using VarianceThreshold (sklearn) takes about 17% of the time (fit takes 7% and transform 10%). 
* **Training :** A total of about 36% of the time is spent on training (combined training time for both stages). About 30% (>80% of the training time) is spend on SVD which is the solver used for QDA.
* **Filtering And Other Pandas Stuff:** Takes about 23% of the time. There are two calls to two different __getitem__ functions -- pandas/core/frame.py and pandas/core/indexing.py.
* **Inference :** Uses predict_proba on the QDA object. This takes a total of ~5% of the time.
* **Misc :** There seems to be a total of about 2-3% time spent on other pandas calls in the final model building. I'm not sure why this doesn't show up in the first round of model building.

## PUBG
### Functionality
* **Read Data:** Just reads the data from the test and train CSV files. Performs some basic filtering. Interestingly, there is a memory reduction function that casts various columns to specific data types. Apparently, the process uses too much memory without this. (TODO is this potentially an opportunity for optimizations like tiling?)
* **Feature Construction:** After reading the CSVs, the algorithm constructs a set of additional features by combining columns in various ways. It roughly has two parts
  * First, a new set of features is constructed by arithmetically combining columns from the input data.
  * Input rows are grouped based on some keys and each group is reduced in different ways (min, max, mean, std etc.). Then the aggregates of each group are appended to each row of the group (using a join).
  * For each different aggregation, the grouping is being re-run. Also, the join is a major bottleneck. If we were to join the aggregate tables first and then perform a single join with the input table, that would be much faster than repeatedly joining the aggregates with the input data.
* **Predictive Models:** There is a single predictive model -- a gradient boosted tree ensemble. The single model is trained with all the augmented training data and then used to predict the test inputs.
### Profiling
* **Read Data:** The read + feature generation takes about 9% of the total time. This time is dominated by the joins (~7%).
 * See above for some possible optimizations.
* **Training:** The training is the biggest bottleneck. Takes about 80% of the time. Almost all of this is spent in the lightgbm.train (~78%).
 *  LightGBM.train seems to have an input called "num_threads". Is this some kind of parallelization?
*  **Inference:** Takes about 10.5% of the total time of which about 7.5% is the actual inference (model.predict) and 3% is reading the test data and adding the derived features.
