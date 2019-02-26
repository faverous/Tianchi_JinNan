from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
boston = pd.read_csv('../data/train_data_forktime.csv')

boston = boston.fillna(-1)
cols = [f for f in boston.columns if f != '收率']
X = boston[cols]
Y = boston["收率"]

## 随即森林特征重要度计算
rf = RandomForestRegressor()
rf.fit(X, Y)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), cols), 
             reverse=True))

## 基于惩罚项的特征选择法