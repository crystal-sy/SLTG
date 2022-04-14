# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:28:06 2022

@author: styra
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

import numpy as np

from config import sltg_config as sltg_config
import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(sltg_config.logging_path)
logger = logging.getLogger('spider')

root_path = 'data/dataset/weibo/'
rf_file= 'result/random_forest/randomForest.pkl'
 
#读入数据集
def loadfile():
    #文件输入
    neg = []
    pos = []
    with open(root_path + "real_test.txt", 'r', encoding='utf-8') as f:
       for line in f.readlines():
           scores = []
           tid, content, comment = line.strip().split("\t")
           if comment != '##':
               scores.append(content)
               scores.append(comment)
               pos.append(scores)
       f.close()
        
    with open(root_path + "fake_test.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            scores = []
            tid, content, comment = line.strip().split("\t")
            if comment != '##':
                scores.append(content)
                scores.append(comment)
                neg.append(scores)
        f.close()
        
    rows = np.concatenate((pos, neg)) #数组拼接
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neg), dtype=int))) #拼接全1和全0数组
    return rows, y

rows, y = loadfile()
x = pd.DataFrame(rows)
#划分数据集
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.2)

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
    model = RandomForestClassifier(n_estimators=200,  max_features='sqrt', 
                                   min_samples_leaf=50, min_samples_split=2, 
                                   bootstrap=True, n_jobs=1, random_state=1)
    
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
