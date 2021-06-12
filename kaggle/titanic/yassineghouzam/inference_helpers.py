import pandas as pd
import numpy as np

def ReadTestData(testCSVFilename):
    test = pd.read_csv(testCSVFilename)
    IDtest = test["PassengerId"]
    # Fill empty and NaNs values with NaN
    test = test.fillna(np.nan)
    return test, IDtest

def ApplyEncoderToColumn(dataset, columnName, encoder):
    colArray = dataset[columnName].values.reshape(-1, 1)
    encodedColumns = encoder.transform(colArray)
    encodedColumnsDF = pd.DataFrame(encodedColumns, index=dataset.index)
    dataset.drop(columns = columnName, inplace=True)
    return pd.concat([dataset, encodedColumnsDF], axis=1)
 
def EngineerFeaturesForInference(dataset, ticketEncoder, embarkedEncoder, cabinEncoder, PclassEncoder, titleEncoder,  age_med, age_pred):

    # When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens.
    # #### Fare

    ## dataset["Fare"].isnull().sum()

    #Fill Fare missing values with the median value
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


    # As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled. 
    # 
    # In this case, it is better to transform it with the log function to reduce this skew. 
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

    # Skewness is clearly reduced after the log transformation

    # ### 3.2 Categorical values
    # #### Sex
    ## train[["Sex","Survived"]].groupby('Sex').mean()

    # It is clearly obvious that Male have less chance to survive than Female.
    # 
    # So Sex, might play an important role in the prediction of the survival.
    # 
    # For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first". 

    # #### Pclass
    # Explore Pclass vs Survived

    # The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.
    # 
    # This trend is conserved when we look at both male and female passengers.

    # #### Embarked
    ## dataset["Embarked"].isnull().sum()

    #Fill Embarked nan values of dataset set with 'S' most frequent value
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

    # Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
    # 
    # However, 1rst class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
    # 
    # Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.

    # convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

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

    for i in index_NaN_age :
        # age_med = dataset["Age"].median()
        # age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred[i]) :
            dataset['Age'].iloc[i] = age_pred[i]
        else :
            dataset['Age'].iloc[i] = age_med

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

    # There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

    # Convert to categorical values Title 
    dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    dataset["Title"] = dataset["Title"].astype(int)

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

    # The family size seems to play an important role, survival probability is worst for large families.
    # 
    # Additionally, i decided to created 4 categories of family size.

    # Create new feature of family size
    dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    # Factorplots of family size categories show that Small and Medium families have more chance to survive than single passenger and large families.
    # convert to indicator values Title and Embarked 
    # dataset = pd.get_dummies(dataset, columns = ["Title"])
    dataset = ApplyEncoderToColumn(dataset, "Title", titleEncoder)
    # dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
    dataset = ApplyEncoderToColumn(dataset, "Embarked", embarkedEncoder)

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

    # Because of the low number of passenger that have a cabin, survival probabilities have an important standard deviation and we can't distinguish between survival probability of passengers in the different desks. 
    # 
    # But we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
    # 
    # It is particularly true for cabin B, C, D, E and F.
    # dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
    dataset = ApplyEncoderToColumn(dataset, "Cabin", cabinEncoder)

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
    dataset = ApplyEncoderToColumn(dataset, "Ticket", ticketEncoder)
    # Create categorical values for Pclass
    # dataset["Pclass"] = dataset["Pclass"].astype("category")
    # dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
    dataset = ApplyEncoderToColumn(dataset, "Pclass", PclassEncoder)

    # Drop useless variables 
    dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
    ## dataset.head()
    return dataset
