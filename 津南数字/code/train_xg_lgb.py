import numpy as np 
import pandas as pd 
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)


train = pd.read_csv('../data/jinnan_round1_train_20181227.csv', encoding = 'utf-8')
test  = pd.read_csv('../data/jinnan_round1_testB_20190121.csv', encoding = 'gb18030')
print(train.shape)
# col = list(train.columns).index('A25')
# row = list(train['样本id']).index('sample_1590')
# print(train.iloc[row,col])
# train.iloc[row,col] = 75
# print(train.iloc[row,col])
######数据预处理######
# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

# cols = ['样本id','B14','A24', 'A10', 'B6', 'B1', 'A27', 'A3', 'A6', 'B8','A5','A7','A9','A11','A14','A16','A26','B5','B7','A20','A28','B4','B9','B10','B11','收率']
# train = train[cols]


# cols = ['样本id','B14','A24', 'A10', 'B6', 'B1', 'A27', 'A3', 'A6', 'B8','A5','A7','A9','A11','A14','A16','A26','B5','B7','A20','A28','B4','B9','B10','B11']
# test = test[cols]

# 删除重复率超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.90:
        good_cols.remove(col)
        print(col,rate)

# 暂时不删除，后面构造特征需要
good_cols.append('A1')
good_cols.append('A3')
good_cols.append('A4')

# 删除异常值
train = train[train['收率']>0.87]
        
train = train[good_cols]
good_cols.remove('收率')
test  = test[good_cols]

best_id = 'simple_908'

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(-1)

def timeTranSecond(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return 7*3600/3600
        elif t=='1900/1/1 2:30':
            return (2*3600+30*60)/3600
        elif t==-1:
            return -1
        else:
            return 0
    
    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    
    return tm
for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f,'应该在前面被删除了！')

def getDuration(se):
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        if se == -1:
            return -1 
        
    try:
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    except:
        if se=='19:-20:05':
            return 1
        elif se=='15:00-1600':
            return 1
    
    return tm
for f in ['A20','A28','B4','B9','B10','B11']:
    try:
        data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
    except:
        print(f, '被删除了')

## 添加样本id作为特征
data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))
# train = data[:train.shape[0]]
# test  = data[train.shape[0]:]
# train['收率'] = target
# train.to_csv('../data/train_data_forktime.csv',index=False)
# test.to_csv('../data/test_data_forktime.csv',index=False)

## 计算所有特征与最有特征组合的差值
# tobest = pd.DataFrame()
# col_names = []
# for i in data.columns:
#     col_name = i+'_best'
#     #data[i] = pd.DataFrame(data[i], dtype=np.float)
#     print(data[i][410])
#     tobest[col_name] = data[i] - data[i][410]
#     col_names.append(col_name)


categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]

# b14/原料总和
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])

numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')

del data['A1']
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A3')
categorical_columns.remove('A4')

# A_tem_sum
data['A_tem_sum'] = data['A6'] + data['A8'] + data['A10'] + data['A12'] + data['A15'] + data['A17'] + data['A21'] + data['A25'] + data['A27']
numerical_columns.append('A_tem_sum')

# A_B_time_sum,
data['A_B_time_sum'] = data['A5']+data['A5']+data['A7']+data['A9']+data['A11']+data['A14']+data['A16']+data['A24']+data['A26']+data['B5']+data['B7']
numerical_columns.append('A_B_time_sum')


#label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
# for f_ in col_names:
#     tobest[f_] = tobest[f_].map(dict(zip(tobest[f_].unique(), range(0, tobest[f_].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]

## 对收率进行分箱
print('*********')
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
mean_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_'+f1+"_"+f2+'_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            print(order_label)
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)             
train.drop(li+['target'], axis=1, inplace=True)

## 对样本id分箱

train['simid'] = pd.cut(train['样本id'], 5, labels=False)
train = pd.get_dummies(train, columns=['simid']) 
li = ['simid_0','simid_1','simid_2','simid_3','simid_4']
mean2_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_'+f1+"_"+f2+'_mean'
            mean2_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean2_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)             
train.drop(li, axis=1, inplace=True) 

print(train.shape)
print(test.shape)

best_col = ['b14/a1_a3_a4_a19_b1_b12', 'A_tem_sum', 'A_B_time_sum', 'B14_to_A5_intTarget_0.0_mean', 'B14_to_A5_intTarget_1.0_mean', 'B14_to_A5_intTarget_2.0_mean', 'B14_to_A5_intTarget_3.0_mean', 'B14_to_A6_intTarget_0.0_mean', 'B14_to_A6_intTarget_1.0_mean', 'B14_to_A6_intTarget_2.0_mean', 'B14_to_A6_intTarget_3.0_mean', 'B14_to_A6_intTarget_4.0_mean', 'B14_to_A7_intTarget_0.0_mean', 'B14_to_A7_intTarget_1.0_mean', 'B14_to_A7_intTarget_2.0_mean', 'B14_to_A7_intTarget_3.0_mean', 'B14_to_A7_intTarget_4.0_mean', 'B14_to_A9_intTarget_0.0_mean', 'B14_to_A9_intTarget_1.0_mean', 'B14_to_A9_intTarget_2.0_mean', 'B14_to_A9_intTarget_3.0_mean', 'B14_to_A9_intTarget_4.0_mean', 'B14_to_A11_intTarget_0.0_mean', 'B14_to_A11_intTarget_1.0_mean', 'B14_to_A11_intTarget_2.0_mean', 'B14_to_A11_intTarget_3.0_mean', 'B14_to_A11_intTarget_4.0_mean', 'B14_to_A14_intTarget_0.0_mean', 'B14_to_A14_intTarget_1.0_mean','B14_to_A14_intTarget_2.0_mean', 'B14_to_A14_intTarget_3.0_mean', 'B14_to_A14_intTarget_4.0_mean', 'B14_to_A16_intTarget_0.0_mean', 'B14_to_A16_intTarget_1.0_mean', 'B14_to_A16_intTarget_2.0_mean', 'B14_to_A16_intTarget_3.0_mean', 'B14_to_A16_intTarget_4.0_mean', 'B14_to_A24_intTarget_0.0_mean', 'B14_to_A24_intTarget_1.0_mean', 'B14_to_A24_intTarget_2.0_mean', 'B14_to_A24_intTarget_3.0_mean', 'B14_to_A24_intTarget_4.0_mean', 'B14_to_A26_intTarget_0.0_mean', 'B14_to_A26_intTarget_1.0_mean', 'B14_to_A26_intTarget_2.0_mean', 'B14_to_A26_intTarget_3.0_mean', 'B14_to_A26_intTarget_4.0_mean', 'B14_to_B1_intTarget_0.0_mean', 'B14_to_B1_intTarget_1.0_mean', 'B14_to_B1_intTarget_2.0_mean', 'B14_to_B1_intTarget_3.0_mean', 'B14_to_B1_intTarget_4.0_mean', 'B14_to_B5_intTarget_0.0_mean', 'B14_to_B5_intTarget_1.0_mean', 'B14_to_B5_intTarget_2.0_mean', 'B14_to_B5_intTarget_3.0_mean', 'B14_to_B5_intTarget_4.0_mean', 'B14_to_B6_intTarget_0.0_mean', 'B14_to_B6_intTarget_1.0_mean', 'B14_to_B6_intTarget_2.0_mean', 'B14_to_B6_intTarget_3.0_mean', 'B14_to_B6_intTarget_4.0_mean', 'B14_to_B7_intTarget_0.0_mean', 'B14_to_B7_intTarget_1.0_mean', 'B14_to_B7_intTarget_2.0_mean', 'B14_to_B7_intTarget_3.0_mean', 'B14_to_B7_intTarget_4.0_mean', 'B14_to_B8_intTarget_0.0_mean', 'B14_to_B8_intTarget_1.0_mean', 'B14_to_B8_intTarget_2.0_mean', 'B14_to_B8_intTarget_3.0_mean', 'B14_to_B8_intTarget_4.0_mean', 'B14_to_B14_intTarget_0.0_mean', 'B14_to_B14_intTarget_1.0_mean', 'B14_to_B14_intTarget_2.0_mean', 'B14_to_B14_intTarget_3.0_mean', 'B14_to_B14_intTarget_4.0_mean', 'B14_to_A5_simid_0_mean', 'B14_to_A5_simid_1_mean', 'B14_to_A5_simid_2_mean', 'B14_to_A5_simid_3_mean', 'B14_to_A5_simid_4_mean', 'B14_to_A6_simid_0_mean', 'B14_to_A6_simid_1_mean', 'B14_to_A6_simid_2_mean', 'B14_to_A6_simid_3_mean', 'B14_to_A6_simid_4_mean', 'B14_to_A7_simid_0_mean', 'B14_to_A7_simid_1_mean', 'B14_to_A7_simid_2_mean', 'B14_to_A7_simid_3_mean', 'B14_to_A7_simid_4_mean', 'B14_to_A9_simid_0_mean', 'B14_to_A9_simid_1_mean', 'B14_to_A9_simid_2_mean', 'B14_to_A9_simid_3_mean', 'B14_to_A9_simid_4_mean', 'B14_to_A11_simid_0_mean', 'B14_to_A11_simid_1_mean', 'B14_to_A11_simid_2_mean', 'B14_to_A11_simid_3_mean', 'B14_to_A11_simid_4_mean', 'B14_to_A14_simid_0_mean', 'B14_to_A14_simid_1_mean', 'B14_to_A14_simid_2_mean', 'B14_to_A14_simid_3_mean', 'B14_to_A14_simid_4_mean', 'B14_to_A16_simid_0_mean', 'B14_to_A16_simid_1_mean', 'B14_to_A16_simid_2_mean', 'B14_to_A16_simid_3_mean', 'B14_to_A16_simid_4_mean', 'B14_to_A24_simid_0_mean', 'B14_to_A24_simid_1_mean', 'B14_to_A24_simid_2_mean', 'B14_to_A24_simid_3_mean', 'B14_to_A24_simid_4_mean', 'B14_to_A26_simid_0_mean', 'B14_to_A26_simid_1_mean', 'B14_to_A26_simid_2_mean', 'B14_to_A26_simid_3_mean', 'B14_to_A26_simid_4_mean', 'B14_to_B1_simid_0_mean', 'B14_to_B1_simid_1_mean', 'B14_to_B1_simid_2_mean', 'B14_to_B1_simid_3_mean', 'B14_to_B1_simid_4_mean', 'B14_to_B5_simid_0_mean', 'B14_to_B5_simid_1_mean', 'B14_to_B5_simid_2_mean', 'B14_to_B5_simid_3_mean', 'B14_to_B5_simid_4_mean', 'B14_to_B6_simid_0_mean', 'B14_to_B6_simid_1_mean', 'B14_to_B6_simid_2_mean', 'B14_to_B6_simid_3_mean', 'B14_to_B6_simid_4_mean', 'B14_to_B7_simid_0_mean', 'B14_to_B7_simid_1_mean', 'B14_to_B7_simid_2_mean', 'B14_to_B7_simid_3_mean', 'B14_to_B7_simid_4_mean', 'B14_to_B8_simid_0_mean', 'B14_to_B8_simid_1_mean', 'B14_to_B8_simid_2_mean', 'B14_to_B8_simid_3_mean', 'B14_to_B8_simid_4_mean', 'B14_to_B14_simid_0_mean', 'B14_to_B14_simid_2_mean', 'B14_to_B14_simid_3_mean', 'B14_to_B14_simid_4_mean']
categorical_columns = ['A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A12', 'A14', 'A15', 'A16', 'A17', 'A19', 'A20', 'A21', 'A22', 'A25', 'A26', 'A27', 'A28', 'B1', 'B4', 'B6', 'B8', 'B9', 'B10', 'B11', 'B12', 'B14']

X_train = train[best_col].values

X_test = test[best_col].values
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
# X_train = sparse.hstack((X_train, tobest[:train.shape[0]]), 'csr')
# X_test = sparse.hstack((X_test, tobest[train.shape[0]:]), 'csr')

print(X_train.shape)
print(X_test.shape)

# best_features = featureSelect(train.columns.tolist(), train, target)
# print(best_features)

######开始训练######
y_train = target.values

param = {'num_leaves': 100,
         'min_data_in_leaf': 20, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.05,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    print(oof_lgb)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

##### xgb
xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
    
print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

#将lgb和xgb的结果进行stacking
train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=10, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 20

# oof_em = np.zeros(len(train))

# for idx in range(len(train)):
#     oof_em[idx] = oof_lgb[idx] * 0.5 + oof_xgb[idx] * 0.5

# predictions = np.zeros(len(test))
# for idx in range(len(test)):
#     predictions[idx] = predictions_lgb[idx] * 0.5 + predictions_xgb[idx] * 0.5
    
print(mean_squared_error(target.values, oof_stack))

testc = pd.read_csv('../data/jinnan_round1_testB_20190121.csv',encoding = 'gb18030')
sub_df = testc[['样本id']]
sub_df[1] = predictions
sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv('../data/result.csv',index=False,header=None)
