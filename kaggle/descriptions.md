# Kaggle Benchmark Descriptions
Descriptions of the benchmarks from Kaggle and some analysis of each of them.
## Instant Gratification
### Functionality
* **Read Data:** Simply reads the data in from the CSV files train and test. No other processing is performed.
* **Predictive Models:**
  * Uses pseudo labeling -- a process where a model is first built and its "confident" predictions are used to generate more training data from test data.
    * Here, any test data predicted to belong to a certain class with probability >= 0.99 is added to the training data.    
  * The first observation is that the column 'wheezy-copper-turtle-magic' takes an integer value between 1 and 512.
  * The algorithm first builds 512 different models based on the value of this column. All the models are [**QuadraticDiscriminantAnalysis**](https://en.wikipedia.org/wiki/Quadratic_classifier).
    * it filters the training and test data based on the value of 'wheezy-copper-turtle-magic' and builds one model for each of these.
### Profiling
