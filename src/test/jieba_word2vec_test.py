# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import jieba
from gensim.models import word2vec
import time
import numpy as np

"""
def get_predict_vecs(words):
    # 加载模型
    wv = word2vec.Word2Vec.load("D:\\hscode\\data\\train\\Word2Vec.pkl")
    #将新的词转换为向量
    train_vecs = build_vector(words,300,wv)
    return train_vecs

def svm_predict(string):
    # 对语句进行分词
    words = jieba.cut(string, cut_all = False)
    # 将分词结果转换为词向量
    word_vecs = get_predict_vecs(words)
    print(word_vecs)

def build_vector(text,size,wv):
    #创建一个指定大小的数据空间
    vec = np.zeros(size).reshape((1,size))
    #count是统计有多少词向量
    count = 0
    #循环所有的词向量进行求和
    for w in text:
        try:
            vec +=  wv[w].reshape((1,size))
            count +=1
        except:
            continue
    #循环完成后求均值
    if count!=0:
        vec/=count
    return vec
    
string = "这本书非常好，我喜欢"

vec = np.zeros(10).reshape((1,10))
svm_predict(string)

"""
s = word2vec.Text8Corpus('D:\\hscode\\data\\demo.txt')
model = word2vec.Word2Vec(s)#这里使用默认的参数训练
s = model.wv.most_similar(u'好', topn = 20)
print(s)
keys = list(model.wv.vocab.keys())
print(keys)
for i in range(len(keys)):
    print(keys[i])
    
s1=model.wv.get_vector(u'好')
print(s1)


text = '赵丽颖主演的正午阳光剧,知否知否应是绿肥红瘦'
seq_list = jieba.cut(text,cut_all=False)
print(seq_list)
print(list(seq_list))

text = 'This is a goog_book.'
seq_list = jieba.cut(text,cut_all=False)
print(seq_list)
print(list(seq_list))



fR = open('D:\\hscode\\data\\xinwen.txt', 'r', encoding='UTF-8')

sent = fR.read()
sent_list = jieba.cut(sent, cut_all=False)


now = int(time.time())     # 1533952277
timeArray = time.localtime(now)
otherStyleTime = time.strftime("%Y%m%d%H%M%S", timeArray)

fW = open('D:\\hscode\\data\\result_' + otherStyleTime + '.txt', 'w', encoding='UTF-8')
fW.write(' '.join(sent_list))

fR.close()
fW.close()





 