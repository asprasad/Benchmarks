import pandas as pd
import numpy as np

#TODO would you really write inference this way -- go over each value of the special feature? Wouldn't
# you rather go over each test input and then pick the appropriate model to use?
def PredictTestData(testDF, models, featureSelectors, cols):
    preds = np.zeros(len(testDF))

    # BUILD 512 SEPARATE MODELS
    for k in range(512):
        test2 = testDF[testDF['wheezy-copper-turtle-magic']==k]
        validTestData = len(test2)!=0
        if not validTestData:
            # print("WARNING_PREDICTION : Zero length test data for "" k: ", k)
            continue

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
        sel = featureSelectors[k]
        test3 = None
        if (validTestData == True):
            test3 = sel.transform(test2[cols])
        
        # STRATIFIED K FOLD
        classifiers = models[k]
        for clf in classifiers:
            if (validTestData == True):
                preds[test2.index] += clf.predict_proba(test3)[:,1] / len(classifiers)
        
    return preds

def PredictTestDataSampleWise(testDF, models, featureSelectors, cols):
    preds = np.zeros(len(testDF))

    for index in testDF.index:
        k = testDF.loc[index]['wheezy-copper-turtle-magic']
        #test2 = testDF[testDF['wheezy-copper-turtle-magic']==k]
        #validTestData = len(test2)!=0
        #if not validTestData:
        #    print("WARNING_PREDICTION : Zero length test data for "" k: ", k)
        #    continue

        sel = featureSelectors[k]
        testVec = testDF.loc[index][cols].values.reshape(1, -1)
        test3 = sel.transform(testVec)
        
        # STRATIFIED K FOLD
        classifiers = models[k]
        for clf in classifiers:
            preds[index] += clf.predict_proba(test3)[:,1] / len(classifiers)
        
        if index%1000 == 0:
            print("Elementwise inference progress -- iteration ", index, "/", len(testDF.index))
        
    return preds

