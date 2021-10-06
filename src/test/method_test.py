# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary test file.
"""

import re
import jieba
import pandas as pd
import numpy as np 
from keras.preprocessing import sequence

"""
def clean_str_sst(string):
    string = re.sub("[，。 :,.；|-“”——_+&;@、《》～（）())#O！：【】\ufeff]", "", string)
    return string.strip().lower()

abc = '《AsadAAAAAasd   》'
print(clean_str_sst(abc))


text = '作者力从马克思注意经济学角度来剖析当代中国经济细心的人会发现中国近20年来'
seq_list = jieba.cut(text,cut_all=False)
print(list(seq_list))
 
w = [] #将所有词语整合在一起  
for i in seq_list:  
  w.append(i)  
  
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
print(dict)
print(len(dict))



dict = pd.Series(w).unique() #统计词的出现次数
print(dict)
print(len(dict))

x_train = [1,2,3,4,5,6,7,8,9]
y_train = [11,12,13,14,15,16,17,18,19]
np.random.seed(4)
np.random.shuffle(x_train) 
np.random.seed(4)
np.random.shuffle(y_train)

print(x_train)
print(y_train)

y_train = [-0.0028714628,-0.0009595467,0.00048220463]
embedding_weights = np.zeros((5, 3)) 
embedding_weights[1, :] = y_train
embedding_weights[2, :] = y_train
print(embedding_weights)

w2dic={}
w2dic['父母'] = 1
w2dic['姐妹'] = 2
print(w2dic)


result_dir = 'D:\\hscode\\result\\'
np.save(result_dir + 'w2dic.npy', w2dic)
np.save(result_dir + 'embedding_weights.npy', embedding_weights)
w2dic_npy = np.load('D:\\hscode\\data\\word2vec\\64\\w2dic.npy').item()
print(len(w2dic_npy))
print(w2dic_npy['父母'])
w2dic_npy = {}


embedding_weights_npy = np.load('D:\\hscode\\data\\word2vec\\64\\embedding_weights.npy')
print(len(embedding_weights_npy))
print(embedding_weights_npy[1188])
embedding_weights_npy = []
"""


data = []
for i in range(10):
    new_txt = []
    for j in range(10):
        new_txt.append(i)
        new_txt.append(j)
    data.append(new_txt)
print(data)

data2 = sequence.pad_sequences(data, maxlen=30)
print(data2)