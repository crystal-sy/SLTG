# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:28:06 2022

@author: styra
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

rf_file='result/random_forest/randomForest.pkl'
 
#读入乳腺癌数据集
data = load_breast_cancer()
rows = []
for row in data.data:
    i = []
    i.append(row[0])
    i.append(row[1])
    rows.append(i)
    
x = pd.DataFrame(rows)
y = data.target
#划分数据集
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

if os.path.exists(rf_file):
    model = joblib.load(rf_file)
else:
    #随机森林，bagging思想
    #n_estimators：随机森林中「树」的数量。
    #max_features：每个分割处的特征数。log2
    #max_depth：每棵树可以拥有的最大「分裂」数。
    #min_samples_split：在树的节点分裂前所需的最少观察数。
    #min_samples_leaf：每棵树末端的叶节点所需的最少观察数。
    #bootstrap：是否使用 bootstrapping 来为随机林中的每棵树提供数据。（bootstrapping 是从数据集中进行替换的随机抽样。）
    model = RandomForestClassifier(n_estimators=1000,  max_features='sqrt', max_depth=None, 
                               min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
    
# 度量随机森林的准确性
model = model.fit(X_train, y_train)
# save model
joblib.dump(model, rf_file) 

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Random Forest train/test accuracies %.3f/%.3f' % (tree_train, tree_test)) 
#RandomFores train/test accuracies 1.000/0.971

#result = model.predict(pd.DataFrame(data.data[0].reshape(1,-1)))
result = model.predict(pd.DataFrame(np.array(rows[0]).reshape(1,-1)))
print(result[0])
