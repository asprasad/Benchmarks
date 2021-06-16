#!/usr/bin/env python
# coding: utf-8

#Source : https://www.kaggle.com/cdeotte/200-magical-models-santander-0-920/

# # The "Magic" of Santander
# In this kernel, we will display pictures of the Santander magic! Previously [here][1], in "Modified Naive Bayes", we saw that we can model each variable separately and then combine the 200 models to score LB 0.899. We will do the same here after adding a "magic" feature to each variable. We will then ensemble the 200 models with logistic regression and score LB 0.920  
#   
# ![image](http://playagricola.com/Kaggle/magic41019.jpg)
#   
# [1]: https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899

# # The "Magic" Feature
# When LGBM "looks" at the histogram for `Var_198`, it "sees" that when `var_198<13` the probability of having `target=1` is high. And when `var_198>13` the probability is low. This can be displayed by showing the predictions made by LGBM (when building a model from only the variable `var_198`). LGBM basically predicts `target=0.18` for `var_198<13` and `target=0.10` otherwise. 
#   
# ![image](http://playagricola.com/Kaggle/198without.png)  
#   
# LGBM "divides" the histogram with **vertical lines** because LGBM does not see **horizontal** differences. A histogram places multiple values into a single bin and produces a smooth picture. If you place every value in its own bin, you will have a jagged picture, where bars change heights from value to value. Some values are unique, some values occur dozens of times (and in the case of `var_108`, some values occur over 300 times!!) Below is a histogram with one value per bin and we zoom in on `11.0000<x<11.1000`. We see that value `11.0712` occurs 5 times and its close neighbor `11.0720` occurs only once.  
#     
# ![image](http://playagricola.com/Kaggle/198zoom3.png)   
#   
# These counts are the "magic" feature. For each variable, we make a new feature (column) whose value is the number of counts of the corresponding variable. An example of this new column is displayed above next to the histogram. When LGBM has this new feature, it can now "divide" the histogram with **horizontal lines** in addition to vertical.  
#   
# ![image](http://playagricola.com/Kaggle/198with.png)  
#   
# Notice now that LGBM predicts `target=0.1` when `var_198<13` AND `count=1`. When  `var_198<13` AND `count>1`, it predicts `target=0.36`. This improvement (using the magic) causes validation AUC to become 0.551 as opposed to 0.547 when using `var_198` alone.

# # Why is the Magic difficult to find?
# The "magic" is difficult to find because the new feature `Var_198_FE` interacts with `Var_198`. Therefore if you add the new feature to an LGBM with `feature_fraction=0.05`, you will not increase your CV or LB. You must set `feature_fraction=1.0`. Then you will gain the benefit from the new feature but you also have the determental effect of modeling spurious original variable interactions. None-the-less, adding the new feature and using `feature_fraction=1.0` achieves CV 0.910. To reach 0.920, we must remove the spurious effects from the original variable interactions.  
#   
# UPDATE: I just discovered another reason why magic is hidden. When calculating frequency counts for the test data, you must remove the **fake** test data before counting. (Fake test described [here][1]) If you don't, then your test predictions will score LB 0.900 instead of LB 0.910 and you may disregard frequency counts as useless.
# 
# # Maximizing Magic Feature
# To maximize the gain of the "magic" feature (and climb from LB 0.910 to 0.920), we must allow the new feature to interact with the original variables while preventing the original variables from interacting with each other. Here are 3 ways to do that:
# * Use Data Augmentation (as shown in Jiwei's awesome kernel [here][2]). You must keep original and new feature in same row.
# * Use 200 separate models as shown in this kernel below.
# * Merge new feature and original feature into one feature. In original data, simply add 200 to each unique value. (And don't add new columns)
# 
# # Let's Begin
# When counting the occurence of each value, we will merge the training data and **real** test data first, and count everything together. In YaG320's brilliant kernel [here][1], we learned that half the test data is fake.  
#   
# [1]: https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
# [2]: https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

# LOAD LIBRARIES
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np, gc
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import statsmodels.api as sm

import cProfile
import joblib
from inference_helpers import ReadTestData, AddMagicFeaturesToTest, DoInference

# GET INDICIES OF REAL TEST DATA FOR FE
#######################
# TAKE FROM YAG320'S KERNEL
# https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split

scriptPath = os.path.realpath(__file__)
scriptDirPath = os.path.dirname(scriptPath)
rootPath = os.path.dirname(scriptDirPath)
dataDirPath = os.path.join(rootPath, "data")

test_path = os.path.join(dataDirPath, 'test.csv')

train = None
test = None
real_samples_indexes = None
synthetic_samples_indexes = None

models = dict()
logisticRegressionModel = None

def ReadData():
    global test, train
    global real_samples_indexes, synthetic_samples_indexes

    df_test = pd.read_csv(test_path)
    df_test.drop(['ID_code'], axis=1, inplace=True)
    df_test = df_test.values

    unique_samples = []
    unique_count = np.zeros_like(df_test)
    for feature in range(df_test.shape[1]):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    print('Found',len(real_samples_indexes),'real test')
    print('Found',len(synthetic_samples_indexes),'fake test')

    ###################

    d = {}
    for i in range(200): d['var_'+str(i)] = 'float32'
    d['target'] = 'uint8'
    d['ID_code'] = 'object'

    train = pd.read_csv(os.path.join(dataDirPath, 'train.csv'), dtype=d)
    test = pd.read_csv(os.path.join(dataDirPath, 'test.csv'), dtype=d)

    print('Loaded',len(train),'rows of train')
    print('Loaded',len(test),'rows of test')

frequencyEncodeMap = dict()

# FREQUENCY ENCODE
def encode_FE(df,col,test):
    global frequencyEncodeMap
    cv = df[col].value_counts()
    nm = col+'_FE'
    frequencyEncodeMap[col] = cv
    df[nm] = df[col].map(cv)
    test[nm] = test[col].map(cv)
    test[nm].fillna(0,inplace=True)
    if cv.max()<=255:
        df[nm] = df[nm].astype('uint8')
        test[nm] = test[nm].astype('uint8')
    else:
        df[nm] = df[nm].astype('uint16')
        test[nm] = test[nm].astype('uint16')        
    return

def AddMagicFeatures():
    global test, train
    test['target'] = -1
    comb = pd.concat([train, test.loc[real_samples_indexes]],axis=0,sort=True)
    for i in range(200): encode_FE(comb,'var_'+ str(i),test)
    train = comb[:len(train)]; del comb
    print('Added 200 new magic features!')

# LGBM PARAMETERS

def DoTrainingAndInference():
    # LGBM PARAMETERS
    param = {
        'learning_rate': 0.04,
        'num_leaves': 3,
        'metric':'auc',
        'boost_from_average':'false',
        'feature_fraction': 1.0,
        'max_depth': -1,
        'objective': 'binary',
        'verbosity': -10}
    train2 = train.sample(frac=1,random_state=42)
    evals_result = {}
    num_vars = 5

    # SAVE OUT-OF-FOLD PREDICTIONS
    all_oof = np.zeros((len(train2),num_vars+1))
    all_oof[:,0] = np.ones(len(train2))
    all_oofB = np.zeros((len(train2),num_vars+1))
    all_oofB[:,0] = np.ones(len(train2))

    # SAVE TEST PREDICTIONS
    all_preds = np.zeros((len(test),num_vars+1))
    all_preds[:,0] = np.ones(len(test))
    all_predsB = np.zeros((len(test),num_vars+1))
    all_predsB[:,0] = np.ones(len(test))

    for j in range(num_vars):
        # MODEL WITH MAGIC
        ModelWithMagic(j, train2, param, evals_result, all_oof, all_preds)

    # # Ensemble 200 Models with LR
    # We now have a model for each variable and its predictions on test and its out-of-fold predictions on train. If we just add (or multiply) the predictions together, the AUC is low. Instead we will use logistic regression to ensemble them. Each set of predictions is a vector of length 200000. We have 200 vectors of out-of-fold predictions, call them `x1, x2, x3, ..., x200`. We know the true train target, call it `y`. We will now use logistic regression to find 200 coefficients (model y from x's). Then we will use those coefficients to combine our 200 test predictions to create a submission.

    # ENSEMBLE MODEL WITH MAGIC
    global logisticRegressionModel
    logr = sm.Logit(train2['target'], all_oof[:,:num_vars+1])
    logr = logr.fit(disp=0)
    ensemble_preds = logr.predict(all_oof[:,:num_vars+1])
    ensemble_auc = roc_auc_score(train2['target'],ensemble_preds)  
    print('##################')
    print('Combined Model with magic Val_AUC=',round(ensemble_auc,5))
    print()

    logisticRegressionModel = logr

    # SAVE PREDICTIONS TO CSV    
    print('Test predictions saved as submission.csv')
    print('OOF predictions saved as oof_submission.csv')
    print('Histogram of test predictions displayed below:')

    sub = train2[['ID_code','target']].copy()
    sub['predict'] = ensemble_preds
    sub.reset_index(inplace=True)
    sub.sort_values('index',inplace=True)
    sub.to_csv(os.path.join(scriptDirPath, 'oof_submission.csv'),index=False)

    test_preds = logr.predict(all_preds[:,:num_vars+1])
    sub = pd.read_csv(os.path.join(dataDirPath, 'sample_submission.csv'))
    sub['target'] = test_preds
    sub.to_csv('submission.csv',index=False)

    joblib.dump(models, os.path.join(scriptDirPath, "models.pkl"))
    joblib.dump(logr, os.path.join(scriptDirPath, "logisticRegressionModel.pkl"))
    joblib.dump(frequencyEncodeMap, os.path.join(scriptDirPath, "frequencyEncodeMap.pkl"))


def ModelWithMagic(j, train2, param, evals_result, all_oof, all_preds):
    global models
    # MODEL WITH MAGIC
    features = ['var_'+ str(j),'var_' + str(j) +'_FE']
    oof = np.zeros(len(train2))
    preds = np.zeros(len(test))

    models[j] = []
    # 5-FOLD WITH MAGIC
    for k in range(5):
            valid = train2.iloc[k*40000:(k+1)*40000]
            train = train2[ ~train2.index.isin(valid.index) ]    
            trn_data  = lgb.Dataset(train[features], label=train['target'])
            val_data = lgb.Dataset(valid[features], label=valid['target'])     
            model = lgb.train(param, trn_data, 750, valid_sets = [trn_data, val_data], 
                    verbose_eval=False, evals_result=evals_result)      
            x = evals_result['valid_1']['auc']
            best = x.index(max(x))
            #print('i=',i,'k=',k,'best=',best)
            oof[k*40000:(k+1)*40000] = model.predict(valid[features], num_iteration=best)
            preds += model.predict(test[features], num_iteration=best)/5.0
            models[j].append((model, best))

    val_auc = roc_auc_score(train2['target'],oof)
    print('VAR_'+str(j)+' with magic val_auc =',round(val_auc,5))
    all_oof[:,j+1] = oof
    all_preds[:,j+1] = preds

def RunTrainingAndInference():
    ReadData()
    AddMagicFeatures()
    DoTrainingAndInference()

def RunInference():
    models = joblib.load(os.path.join(scriptDirPath, "models.pkl"))
    logisticRegressionModel = joblib.load(os.path.join(scriptDirPath, "logisticRegressionModel.pkl"))
    frequencyEncodeMap = joblib.load(os.path.join(scriptDirPath, "frequencyEncodeMap.pkl"))

    test = ReadTestData(os.path.join(dataDirPath, 'test.csv'))
    AddMagicFeaturesToTest(test, frequencyEncodeMap)
    DoInference(test, models, logisticRegressionModel, 5, dataDirPath)

runTraining = False
if runTraining:
    RunTrainingAndInference()

profileInference = True
if profileInference:
    cProfile.run("RunInference()", filename=os.path.join(os.path.dirname(scriptPath), os.path.basename(__file__) + ".prof"))
else:
    RunInference()
