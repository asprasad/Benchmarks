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
