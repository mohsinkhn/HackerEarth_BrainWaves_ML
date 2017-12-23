#import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from  sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline, make_union
import lightgbm as lgb
import category_encoders as en
from TargetEncoder import TargetEncoder
import pickle
import time
import os
import warnings
warnings.simplefilter("ignore")


from FeatureSubsetTransformer import FeatureSubsetTransformer
##############################################################################
#useful functions
def predict_proba_corr(self, X):
    preds = self.predict_proba(X)[:,1]
    d0 = 0.2
    d1 = 1 - d0
    r0 = np.mean(preds)
    r1 = 1 - r0
    gamma_0 = r0/d0
    gamma_1 = r1/d1
    return gamma_1*preds/(gamma_1*preds + gamma_0*(1 - preds))
##############################################################################


#Read data with features
print("Reading data with features")
train = pd.read_csv("../utility/train.csv")
test = pd.read_csv("../utility/test.csv")

#Set up pipeline
feats = [f for f in train.columns if f not in ['transaction_id', 'target']]
X = train[feats]
y = train['target']
X_test = test[feats]

cvlist = list(StratifiedKFold(n_splits=5, random_state=5).split(X, y))

lgb_params = {"learning_rate": 0.003,
              'metric': 'auc',
              'n_estimators': 1000,
              #'num_boost_round': 20,
              'max_depth': 8,
              'reg_lambds': 0.0005,
              'min_data_in_leaf': 100,
              'feature_fraction': 0.8,
              'bagging_fraction': 0.8,
              'objective': 'binary',
              'num_leaves': 256,
              'verbose': 10
             }

Pipeline.predict_proba_corr = predict_proba_corr

cat_feats = [c for c in feats if 'cat_' in c]

fs_sub = FeatureSubsetTransformer(feats)

en_target = TargetEncoder(cols=cat_feats, add_to_orig=True)
en_onehot = en.OneHotEncoder(cols=cat_feats)
en_binary = en.BinaryEncoder(cols=cat_feats)

for enc in ['target', 'onehot', 'en_binary', 'none']:
    lgb1 = lgb.LGBMClassifier(**lgb_params)
    feat_name = "base_feats" + '_' + enc
    if enc == 'target':
        print("here.......................")
        pipe = Pipeline([(feat_name,
                   make_pipeline(fs_sub, en_target)), 
             ('lgb', lgb1)])
    elif enc == 'binary':
        pipe = Pipeline([(feat_name,
                   make_pipeline(fs_sub, en_binary)), 
             ('lgb', lgb1)])
    elif enc == 'onehot':
        pipe = Pipeline([(feat_name,
                   make_pipeline(fs_sub, en_onehot)), 
             ('lgb', lgb1)])
    else: 
        pipe = Pipeline([(feat_name,fs_sub), 
             ('lgb', lgb1)])
    print("Encoding...........",enc)
    print("Getting predictions on train set")
    oof_preds = cross_val_predict(pipe, X, y, cv=cvlist, verbose=10, method='predict_proba_corr')	
    print("ROC AUC SCORE on out f fold predictions",roc_auc_score(y, oof_preds))                                          
    print("Getting predictions on test set")
    ests = int(lgb_params['n_estimators']/( 1 -  1/len(cvlist)))
    lgb1 = lgb.LGBMClassifier(**lgb_params).set_params(n_estimators=ests)
    test_preds = pipe.fit(X,y).predict_proba_corr(X_test)   

    print("Writing out oof preds")
    filename_train = '../utility/' + '_'.join(list(pipe.named_steps.keys())) + 'oofpreds.csv'
    preds_df = pd.DataFrame({'id': train.transaction_id, 'target':oof_preds})
    preds_df.to_csv(filename_train, index=False)

    print("Writing test preds")
    print(len(test_preds))
    filename_test = '../utility/' + '_'.join(list(pipe.named_steps.keys())) + 'testpreds.csv'
    preds_test_df = pd.DataFrame({'id': test.transaction_id, 'target':test_preds})
    preds_test_df.to_csv(filename_test, index=False)                                

