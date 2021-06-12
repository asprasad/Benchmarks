#!/usr/bin/env python
# coding: utf-8

#Source : https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/data?select=train.csv

# # Titanic Top 4% with ensemble modeling
# ### **Yassine Ghouzam, PhD**
# #### 13/07/2017
# 
# * **1 Introduction**
# * **2 Load and check data**
#     * 2.1 load data
#     * 2.2 Outlier detection
#     * 2.3 joining train and test set
#     * 2.4 check for null and missing values
# * **3 Feature analysis**
#     * 3.1 Numerical values
#     * 3.2 Categorical values
# * **4 Filling missing Values**
#     * 4.1 Age
# * **5 Feature engineering**
#     * 5.1 Name/Title
#     * 5.2 Family Size
#     * 5.3 Cabin
#     * 5.4 Ticket
# * **6 Modeling**
#     * 6.1 Simple modeling
#         * 6.1.1 Cross validate models
#         * 6.1.2 Hyperparamater tunning for best models
#         * 6.1.3 Plot learning curves
#         * 6.1.4 Feature importance of the tree based classifiers
#     * 6.2 Ensemble modeling
#         * 6.2.1 Combining models
#     * 6.3 Prediction
#         * 6.3.1 Predict and Submit results
#     

# ## 1. Introduction
# 
# This is my first kernel at Kaggle. I choosed the Titanic competition which is a good way to introduce feature engineering and ensemble modeling. Firstly, I will display some feature analyses then ill focus on the feature engineering. Last part concerns modeling and predicting the survival on the Titanic using an voting procedure. 
# 
# This script follows three main parts:
# 
# * **Feature analysis**
# * **Feature engineering**
# * **Modeling**

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns
import cProfile
# get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from inference_helpers import EngineerFeaturesForInference, ReadTestData

sns.set(style='white', context='notebook', palette='deep')


# ## 2. Load and check data
# ### 2.1 Load data

train = None
test = None
IDTest = None
dataset = None
X_train = None
Y_train = None
plotGraphs = False
train_len = -1

scriptPath = os.path.realpath(__file__)
scriptDirPath = os.path.dirname(scriptPath)
rootPath = os.path.dirname(scriptDirPath)
dataDirPath = os.path.join(rootPath, "data")


# Load data
##### Load train and Test set
def ReadData():
    global train, test, IDTest
    train = pd.read_csv(os.path.join(dataDirPath, "train.csv"))
    test = pd.read_csv(os.path.join(dataDirPath, "test.csv"))
    IDtest = test["PassengerId"]

# ### 2.2 Outlier detection

# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


def ProcessData():
    global test, train, dataset, train_len
    # detect outliers from Age, SibSp , Parch and Fare
    Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


    # Since outliers can have a dramatic effect on the prediction (espacially for regression problems), i choosed to manage them. 
    # 
    # I used the Tukey method (Tukey JW., 1977) to detect ouliers which defines an interquartile range comprised between the 1st and 3rd quartile of the distribution values (IQR). An outlier is a row that have a feature value outside the (IQR +- an outlier step).
    # 
    # 
    # I decided to detect outliers from the numerical values features (Age, SibSp, Sarch and Fare). Then, i considered outliers as rows that have at least two outlied numerical values.

    train.loc[Outliers_to_drop] # Show the outliers rows


    # We detect 10 outliers. The 28, 89 and 342 passenger have an high Ticket Fare 
    # 
    # The 7 others have very high values of SibSP.
    # Drop outliers
    train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


    # ### 2.3 joining train and test set

    ## Join train and test datasets in order to obtain the same number of features during categorical conversion
    train_len = len(train)
    dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


    # I join train and test datasets to obtain the same number of features during categorical conversion (See feature engineering).

    # ### 2.4 check for null and missing values

    # In[7]:


    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)

    # Check for Null values
    ## dataset.isnull().sum()

    # Age and Cabin features have an important part of missing values.
    # 
    # **Survived missing values correspond to the join testing dataset (Survived column doesn't exist in test set and has been replace by NaN values when concatenating the train and test set)**

    # Infos
    # train.info()
    # train.isnull().sum()
    # train.head()
    # train.dtypes

    ### Summarize data
    # Summarie and statistics
    # train.describe()

usePandasForDummyEncode = False

def DummyEncodeColumn(dataset, columnName):
    if usePandasForDummyEncode == True:
        return pd.get_dummies(dataset, columns=[columnName]), None

    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    colArray = dataset[columnName].values.reshape(-1, 1)
    ohe.fit(colArray)
    encodedColumns = ohe.transform(colArray)
    encodedColumnsDF = pd.DataFrame(encodedColumns, index=dataset.index)
    dataset.drop(columns = columnName, inplace=True)
    return (pd.concat([dataset, encodedColumnsDF], axis=1), ohe)

ticketEncoder = None
embarkedEncoder = None
cabinEncoder = None
PclassEncoder = None
titleEncoder = None 
age_med = 0
age_pred = dict()
# ## 3. Feature analysis
# ### 3.1 Numerical values

def FeatureAnalysis():
    global test, train, dataset, X_train, Y_train, plotGraphs
    if plotGraphs:
        # Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
        g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

        # Only Fare feature seems to have a significative correlation with the survival probability.
        # 
        # It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

        # #### SibSP
        # Explore SibSp feature vs Survived
        g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
        palette = "muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")

        # It seems that passengers having a lot of siblings/spouses have less chance to survive
        # 
        # Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive
        # 
        # This observation is quite interesting, we can consider a new feature describing these categories (See feature engineering)
        # #### Parch
        # Explore Parch feature vs Survived
        g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 
        palette = "muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")

        # Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).
        # 
        # Be carefull there is an important standard deviation in the survival of passengers with 3 parents/children 
        # #### Age
        # Explore Age vs Survived
        g = sns.FacetGrid(train, col='Survived')
        g = g.map(sns.distplot, "Age")

        # Age distribution seems to be a tailed distribution, maybe a gaussian distribution.
        # 
        # We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived. 
        # 
        # So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
        # 
        # It seems that very young passengers have more chance to survive.
        # Explore Age distibution 
        g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
        g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
        g.set_xlabel("Age")
        g.set_ylabel("Frequency")
        g = g.legend(["Not Survived","Survived"])

    # When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens.
    # #### Fare

    ## dataset["Fare"].isnull().sum()

    #Fill Fare missing values with the median value
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

    if plotGraphs:
        # Since we have one missing value , i decided to fill it with the median value which will not have an important effect on the prediction.
        # Explore Fare distribution 
        g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
        g = g.legend(loc="best")

    # As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled. 
    # 
    # In this case, it is better to transform it with the log function to reduce this skew. 
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    if plotGraphs:
        g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
        g = g.legend(loc="best")

    # Skewness is clearly reduced after the log transformation

    # ### 3.2 Categorical values
    # #### Sex
    if plotGraphs:
        g = sns.barplot(x="Sex",y="Survived",data=train)
        g = g.set_ylabel("Survival Probability")

    ## train[["Sex","Survived"]].groupby('Sex').mean()

    # It is clearly obvious that Male have less chance to survive than Female.
    # 
    # So Sex, might play an important role in the prediction of the survival.
    # 
    # For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first". 

    # #### Pclass
    # Explore Pclass vs Survived
    if plotGraphs:
        g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 
        palette = "muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")

        # Explore Pclass vs Survived by Sex
        g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                        size=6, kind="bar", palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")


    # The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.
    # 
    # This trend is conserved when we look at both male and female passengers.

    # #### Embarked
    ## dataset["Embarked"].isnull().sum()

    #Fill Embarked nan values of dataset set with 'S' most frequent value
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

    if plotGraphs:
        # Since we have two missing values , i decided to fill them with the most fequent value of "Embarked" (S).
        # Explore Embarked vs Survived 
        g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                        size=6, kind="bar", palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("survival probability")


        # It seems that passenger coming from Cherbourg (C) have more chance to survive.
        # 
        # My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
        # 
        # Let's see the Pclass distribution vs Embarked

        # Explore Pclass vs Embarked 
        g = sns.factorplot("Pclass", col="Embarked",  data=train,
                        size=6, kind="count", palette="muted")
        g.despine(left=True)
        g = g.set_ylabels("Count")

        # Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.
        # 
        # At this point, i can't explain why first class has an higher survival rate. My hypothesis is that first class passengers were prioritised during the evacuation due to their influence.

        # ## 4. Filling missing Values
        # ### 4.1 Age
        # 
        # As we see, Age column contains 256 missing values in the whole dataset.
        # 
        # Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values. 
        # 
        # To adress this problem, i looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).
        # Explore Age vs Sex, Parch , Pclass and SibSP
        g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
        g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
        g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
        g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")


    # Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
    # 
    # However, 1rst class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
    # 
    # Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.

    # convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

    if plotGraphs:
        g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)

    # The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
    # 
    # In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
    # 
    # So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
    # 
    # The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

    # Filling missing value of Age 

    ## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
    # Index of NaN age rows
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
    global age_med, age_pred
    for i in index_NaN_age :
        age_med = dataset["Age"].median()
        age_pred[i - train_len] = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred[i - train_len]) :
            dataset['Age'].iloc[i] = age_pred[i - train_len]
        else :
            dataset['Age'].iloc[i] = age_med

    if plotGraphs:
        g = sns.factorplot(x="Survived", y = "Age",data = train, kind="box")
        g = sns.factorplot(x="Survived", y = "Age",data = train, kind="violin")

    # No difference between median value of age in survived and not survived subpopulation. 
    # 
    # But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

    # ## 5. Feature engineering
    # ### 5.1 Name/Title

    ## dataset["Name"].head()

    # The Name feature contains information on passenger's title.
    # 
    # Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.

    # Get Title from Name
    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    ## dataset["Title"].head()

    if plotGraphs:
        g = sns.countplot(x="Title",data=dataset)
        g = plt.setp(g.get_xticklabels(), rotation=45) 

    # There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

    # Convert to categorical values Title 
    dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    dataset["Title"] = dataset["Title"].astype(int)

    if plotGraphs:
        g = sns.countplot(dataset["Title"])
        g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])

        g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
        g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
        g = g.set_ylabels("survival probability")

    # "Women and children first" 
    # 
    # It is interesting to note that passengers with rare title have more chance to survive.
    # Drop Name variable
    dataset.drop(labels = ["Name"], axis = 1, inplace = True)

    # ### 5.2 Family size
    # 
    # We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, i choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger).

    # Create a family size descriptor from SibSp and Parch
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

    if plotGraphs:
        g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
        g = g.set_ylabels("Survival Probability")

    # The family size seems to play an important role, survival probability is worst for large families.
    # 
    # Additionally, i decided to created 4 categories of family size.

    # Create new feature of family size
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    if plotGraphs:
        g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
        g = g.set_ylabels("Survival Probability")
        g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
        g = g.set_ylabels("Survival Probability")
        g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
        g = g.set_ylabels("Survival Probability")
        g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
        g = g.set_ylabels("Survival Probability")

    # Factorplots of family size categories show that Small and Medium families have more chance to survive than single passenger and large families.
    # convert to indicator values Title and Embarked 
    
    # dataset = pd.get_dummies(dataset, columns = ["Title"])
    global titleEncoder
    dataset, titleEncoder = DummyEncodeColumn(dataset, "Title")
    # dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
    global embarkedEncoder
    dataset, embarkedEncoder = DummyEncodeColumn(dataset, "Embarked")

    ## dataset.head()

    # At this stage, we have 22 features.
    # ### 5.3 Cabin
    ## dataset["Cabin"].head()
    ## dataset["Cabin"].describe()
    ## dataset["Cabin"].isnull().sum()

    # The Cabin feature column contains 292 values and 1007 missing values.
    # 
    # I supposed that passengers without a cabin have a missing value displayed instead of the cabin number.
    ## dataset["Cabin"][dataset["Cabin"].notnull()].head()

    # Replace the Cabin number by the type of cabin 'X' if not
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])

    if plotGraphs:
        # The first letter of the cabin indicates the Desk, i choosed to keep this information only, since it indicates the probable location of the passenger in the Titanic.
        g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])

        g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
        g = g.set_ylabels("Survival Probability")

    # Because of the low number of passenger that have a cabin, survival probabilities have an important standard deviation and we can't distinguish between survival probability of passengers in the different desks. 
    # 
    # But we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
    # 
    # It is particularly true for cabin B, C, D, E and F.
    # dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
    global cabinEncoder
    dataset, cabinEncoder = DummyEncodeColumn(dataset, "Cabin")

    # ### 5.4 Ticket
    ## dataset["Ticket"].head()

    # It could mean that tickets sharing the same prefixes could be booked for cabins placed together. It could therefore lead to the actual placement of the cabins within the ship.
    # 
    # Tickets with same prefixes may have a similar class and survival.
    # 
    # So i decided to replace the Ticket feature column by the ticket prefixe. Which may be more informative.
    ## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit() :
            Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
        else:
            Ticket.append("X")
            
    dataset["Ticket"] = Ticket
    ## dataset["Ticket"].head()

    # dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
    global ticketEncoder
    dataset, ticketEncoder = DummyEncodeColumn(dataset, "Ticket")
    # Create categorical values for Pclass
    # dataset["Pclass"] = dataset["Pclass"].astype("category")
    # dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
    global PclassEncoder
    dataset, PclassEncoder = DummyEncodeColumn(dataset, "Pclass")

    # Drop useless variables 
    dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
    ## dataset.head()

def InitTrainingData():
    global train, test, dataset, train_len, X_train, Y_train
    # ## 6. MODELING
    ## Separate train dataset and test dataset

    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=["Survived"],axis = 1,inplace=True)

    ## Separate train features and label 
    train["Survived"] = train["Survived"].astype(int)
    Y_train = train["Survived"]
    X_train = train.drop(labels = ["Survived"],axis = 1)

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

def EvaluateSimpleModels():
    global X_train, Y_train
    # ### 6.1 Simple modeling
    # #### 6.1.1 Cross validate models
    # 
    # I compared 10 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
    # 
    # * SVC
    # * Decision Tree
    # * AdaBoost 
    # * Random Forest
    # * Extra Trees
    # * Gradient Boosting
    # * Multiple layer perceprton (neural network)
    # * KNN
    # * Logistic regression
    # * Linear Discriminant Analysis

    # Modeling step Test differents algorithms 
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state = random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    cv_results = []
    for classifier in classifiers :
        cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
    "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

    if plotGraphs:
        g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
        g.set_xlabel("Mean Accuracy")
        g = g.set_title("Cross validation scores")


# I decided to choose the SVC, AdaBoost, RandomForest , ExtraTrees and the GradientBoosting classifiers for the ensemble modeling.

# #### 6.1.2 Hyperparameter tunning for best models
# 
# I performed a grid search optimization for AdaBoost, ExtraTrees , RandomForest, GradientBoosting and SVC classifiers.
# 
# I set the "n_jobs" parameter to 4 since i have 4 cpu . The computation time is clearly reduced.
# 
# But be carefull, this step can take a long time, i took me 15 min in total on 4 cpu.

### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING
ada_best = None
def EvaluateADABoost():
    global ada_best, X_train, Y_train
    # Adaboost
    DTC = DecisionTreeClassifier()

    adaDTC = AdaBoostClassifier(DTC, random_state=7)

    ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                "base_estimator__splitter" :   ["best", "random"],
                "algorithm" : ["SAMME","SAMME.R"],
                "n_estimators" :[1,2],
                "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

    gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
    gsadaDTC.fit(X_train,Y_train)
    ada_best = gsadaDTC.best_estimator_
    print("ADABoost best score : ", gsadaDTC.best_score_)

ExtC_best = None
def EvaluateExtraTreesClassifier():
    global ExtC_best, X_train, Y_train
    #ExtraTrees 
    ExtC = ExtraTreesClassifier()

    ## Search grid for optimal parameters
    ex_param_grid = {"max_depth": [None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [False],
                "n_estimators" :[100,300],
                "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsExtC.fit(X_train,Y_train)

    ExtC_best = gsExtC.best_estimator_

    # Best score
    print("Extra tree classifier best score : ", gsExtC.best_score_)

RFC_best = None
def EvaluateRandomForest():
    global RFC_best, X_train, Y_train
    # RFC Parameters tunning 
    RFC = RandomForestClassifier()

    ## Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                "max_features": [1, 3, 10],
                "min_samples_split": [2, 3, 10],
                "min_samples_leaf": [1, 3, 10],
                "bootstrap": [False],
                "n_estimators" :[100,300],
                "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsRFC.fit(X_train,Y_train)

    RFC_best = gsRFC.best_estimator_

    # Best score
    print("RandomForest best score : ", gsRFC.best_score_)

GBC_best = None
def EvaluateGradientBoosting():
    global GBC_best, X_train, Y_train
    # Gradient boosting tunning
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ["deviance"],
                'n_estimators' : [100,200,300],
                'learning_rate': [0.1, 0.05, 0.01],
                'max_depth': [4, 8],
                'min_samples_leaf': [100,150],
                'max_features': [0.3, 0.1] 
                }

    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsGBC.fit(X_train,Y_train)

    GBC_best = gsGBC.best_estimator_

    # Best score
    print("GradientBoost best score : ", gsGBC.best_score_)

SVMC_best = None
def EvaluateSVC():
    global SVMC_best, X_train, Y_train
    ### SVC classifier
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'], 
                    'gamma': [ 0.001, 0.01, 0.1, 1],
                    'C': [1, 10, 50, 100,200,300, 1000]}

    gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

    gsSVMC.fit(X_train,Y_train)

    SVMC_best = gsSVMC.best_estimator_

    # Best score
    print("SVC best score : ",gsSVMC.best_score_)


# #### 6.1.3 Plot learning curves
# 
# Learning curves are a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    if not plotGraphs:
        return

    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def PlotLearningCurves():
    if not plotGraphs:
        return
    g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
    g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
    g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
    g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
    g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)

    # GradientBoosting and Adaboost classifiers tend to overfit the training set. According to the growing cross-validation curves GradientBoosting and Adaboost could perform better with more training examples.
    # 
    # SVC and ExtraTrees classifiers seem to better generalize the prediction since the training and cross-validation curves are close together.

    # #### 6.1.4 Feature importance of tree based classifiers
    # 
    # In order to see the most informative features for the prediction of passengers survival, i displayed the feature importance for the 4 tree based classifiers.
    nrows = ncols = 2
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

    names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            name = names_classifiers[nclassifier][0]
            classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
            g.set_xlabel("Relative importance",fontsize=12)
            g.set_ylabel("Features",fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            nclassifier += 1


# I plot the feature importance for the 4 tree based classifiers (Adaboost, ExtraTrees, RandomForest and GradientBoosting).
# 
# We note that the four classifiers have different top features according to the relative importance. It means that their predictions are not based on the same features. Nevertheless, they share some common important features for the classification , for example 'Fare', 'Title_2', 'Age' and 'Sex'.
# 
# Title_2 which indicates the Mrs/Mlle/Mme/Miss/Ms category is highly correlated with Sex.
# 
# We can say that: 
# 
# - Pc_1, Pc_2, Pc_3 and Fare refer to the general social standing of passengers.
# 
# - Sex and Title_2 (Mrs/Mlle/Mme/Miss/Ms) and Title_3 (Mr) refer to the gender.
# 
# - Age and Title_1 (Master) refer to the age of passengers.
# 
# - Fsize, LargeF, MedF, Single refer to the size of the passenger family.
# 
# **According to the feature importance of this 4 classifiers, the prediction of the survival seems to be more associated with the Age, the Sex, the family size and the social standing of the passengers more than the location in the boat.**
test_Survived = None
votingC = None
def EvaluateVotingClassifier():
    global IDTest, X_train, Y_train, votingC
    IDTest_local = IDTest
    
    if plotGraphs:
        test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
        test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
        test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
        test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
        test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")
        # Concatenate all classifier results
        ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)
        g= sns.heatmap(ensemble_results.corr(),annot=True)
    # The prediction seems to be quite similar for the 5 classifiers except when Adaboost is compared to the others classifiers.
    # 
    # The 5 classifiers give more or less the same prediction but there is some differences. Theses differences between the 5 classifier predictions are sufficient to consider an ensembling vote. 
    # ### 6.2 Ensemble modeling
    # #### 6.2.1 Combining models
    # 
    # I choosed a voting classifier to combine the predictions coming from the 5 classifiers.
    # 
    # I preferred to pass the argument "soft" to the voting parameter to take into account the probability of each vote.
    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
    ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
    votingC = votingC.fit(X_train, Y_train)
    # ### 6.3 Prediction
    # #### 6.3.1 Predict and Submit results
    global test_Survived
    test_Survived = pd.Series(votingC.predict(test), name="Survived")
    results = test_Survived #pd.concat([IDtest, test_Survived],axis=1)
    results.to_csv(os.path.join(scriptDirPath, "ensemble_python_voting.csv"),index=False)

def VotingClassifierInference(testData):
    global votingC, test
    # IDTest_local = IDTest
    
    # ### 6.3 Prediction
    # #### 6.3.1 Predict and Submit results
    test_prediction = pd.Series(votingC.predict(testData), name="Survived")
    results = test_prediction #pd.concat([IDtest, test_Survived],axis=1)
    # print ("Test shape (original): ", test.shape)
    # print ("Test data shape (computed for inference) : ", testData.shape)
    # colsEqual = (test.reset_index(drop=True) == testData.reset_index(drop=True)).all()
    # for c in colsEqual:
    #     print(c)
    # print("Predictions equal : ", (test_Survived == test_prediction).all())
    results.to_csv(os.path.join(scriptDirPath, "ensemble_python_voting.csv"),index=False)

def RunEvaluations():
    ReadData()
    ProcessData()
    FeatureAnalysis()
    InitTrainingData()
    EvaluateSimpleModels()
    EvaluateADABoost()
    EvaluateExtraTreesClassifier()
    EvaluateRandomForest()
    EvaluateGradientBoosting()
    EvaluateSVC()
    EvaluateVotingClassifier()

# TODO This assumes that we've run all the Evaluations before. Also, we're reading the whole of the test and train data and
# manipaulting the whole DF. Separating out the one-hot encoding is complicated and needs to be done. For now, this is just 
# to get some rough idea of the relative costs. 
def RunInferenceOnly():
    # ReadData()
    # ProcessData()
    testData, testID = ReadTestData(os.path.join(dataDirPath, 'test.csv'))
    testData = EngineerFeaturesForInference(testData, ticketEncoder, embarkedEncoder, cabinEncoder, PclassEncoder, titleEncoder, age_med, age_pred)
    # InitTrainingData()
    VotingClassifierInference(testData)

# If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated :)

RunEvaluations()
profileInference = True
if profileInference == True:
    cProfile.run("RunInferenceOnly()", filename=os.path.join(os.path.dirname(scriptPath), os.path.basename(__file__) + ".prof"))
else:
    RunInferenceOnly()