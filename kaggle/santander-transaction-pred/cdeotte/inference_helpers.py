import pandas as pd
import numpy as np
import os

def ReadTestData(test_path):
    d = {}
    for i in range(200): d['var_'+str(i)] = 'float32'
    d['target'] = 'uint8'
    d['ID_code'] = 'object'

    test = pd.read_csv(test_path, dtype=d)

    print('Loaded',len(test),'rows of test')
    return test

def encode_FE(col, test, frequencyEncodeMap):
    cv = frequencyEncodeMap[col]
    nm = col+'_FE'
    test[nm] = test[col].map(cv)
    test[nm].fillna(0,inplace=True)
    if cv.max()<=255:
        test[nm] = test[nm].astype('uint8')
    else:
        test[nm] = test[nm].astype('uint16')        
    return

def AddMagicFeaturesToTest(test, frequencyEncodeMap):
    test['target'] = -1
    for i in range(200):
        encode_FE('var_'+ str(i), test, frequencyEncodeMap)
    print('Added 200 new magic features to test set!')

def DoInference(test, models, logr, num_vars, dataDirPath):
    evals_result = {}

    # SAVE TEST PREDICTIONS
    all_preds = np.zeros((len(test),num_vars+1))
    all_preds[:,0] = np.ones(len(test))

    for j in range(num_vars):
        # MODEL WITH MAGIC
        ModelWithMagic(j, models, test, all_preds)

    # # Ensemble 200 Models with LR
    # We now have a model for each variable and its predictions on test and its out-of-fold predictions on train. If we just add (or multiply) the predictions together, the AUC is low. Instead we will use logistic regression to ensemble them. Each set of predictions is a vector of length 200000. We have 200 vectors of out-of-fold predictions, call them `x1, x2, x3, ..., x200`. We know the true train target, call it `y`. We will now use logistic regression to find 200 coefficients (model y from x's). Then we will use those coefficients to combine our 200 test predictions to create a submission.

    # ENSEMBLE MODEL WITH MAGIC

    # SAVE PREDICTIONS TO CSV    
    print('Test predictions saved as submission.csv')

    test_preds = logr.predict(all_preds[:,:num_vars+1])
    sub = pd.read_csv(os.path.join(dataDirPath, 'sample_submission.csv'))
    sub['target'] = test_preds
    sub.to_csv('submission.csv',index=False)


def ModelWithMagic(j, models, test, all_preds):
    # MODEL WITH MAGIC
    features = ['var_'+ str(j), 'var_' + str(j) +'_FE']
    preds = np.zeros(len(test))

    # 5-FOLD WITH MAGIC
    modelList = models[j]
    for model, best in modelList:
        preds += model.predict(test[features], num_iteration=best)/5.0

    all_preds[:,j+1] = preds


