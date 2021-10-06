# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:53:29 2021

@author: ~styra~
训练词向量
"""

"""
numpy是python扩展程序库，支持大量的维度数组与矩阵运算，
此外也针对数组运算提供大量的数学函数库
"""
import numpy as np 
# jieba分词
import jieba
# 自然语言处理NLP神器--gensim，词向量Word2Vec
from gensim.models.word2vec import Word2Vec

import sys
# multiprocessing包是Python中的多进程管理包。 与threading.Thread类似,
# 它可以利用multiprocessing.Process对象来创建一个进程。
import multiprocessing
import time
import os
from gensim.models import KeyedVectors

# CPU运行
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 设置递归深度
sys.setrecursionlimit(1000000)

# 使得随机数据可预测
np.random.seed()

# 参数配置
cpu_count = multiprocessing.cpu_count() - 4  # 4CPU数量
voc_dim = 128 # word的向量维度
min_out = 3 # 单词出现次数
window_size = 5 # WordVec中的滑动窗口大小
data_dir = 'D:\\hscode\\data\\'
result_dir = 'D:\\hscode\\result\\'

def loadfile():
    #文件输入
    neg = []
    pos = []
    with open(data_dir + 'pos.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            pos.append(line)
        f.close()
    with open(data_dir + 'neg.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            neg.append(line)
        f.close()
    X_Vec = np.concatenate((pos, neg)) #数组拼接
    return X_Vec

def file_jieba_cut(text):
    now = int(time.time())
    timeArray = time.localtime(now)
    nowTime = time.strftime("%Y%m%d%H%M%S", timeArray)
    
    result=[]
    for document in text:
        result.append(jiebacut(document.replace('\n', ''), nowTime) )
    return result

def jiebacut(text, nowTime):
    # 将语句分词
    ret = [];
    sent_list = jieba.cut(text, cut_all = False) # 精确模式
    ret = list(sent_list)
    
    """
    # 追加写入
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fW = open(result_dir + 'result_' + nowTime + '.txt', 'a', encoding='UTF-8')
    fW.write(' '.join(ret))
    fW.write('\n')
    fW.close()
    """
    return ret

def word2vec_train(X_Vec):
    #size:是指特征向量的维度,默认为100.大的size需要更多的训练数据,但是效果会更好.推荐值为几十到几百
    #min_count:可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    #window:表示当前词与预测词在一个句子中的最大距离是多少
    #workers:参数控制训练的并行数。
    #iter:迭代次数，默认为5
    """
    该构造函数执行了三个步骤：建立一个空的模型对象，遍历一次语料库建立词典，
    第二次遍历语料库建立神经网络模型可以通过分别执行
    model=gensim.models.Word2Vec()，model.build_vocab(sentences)，
    model.train(sentences)来实现
    """
    
    model_word = Word2Vec(size=voc_dim,
                        min_count=min_out,
                        window=window_size,
                        workers=cpu_count,
                        iter=5)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.epochs)
    
    dict = list(model_word.wv.vocab.keys())
    input_dim = len(dict) + 1 #下标0空出来给不够10的字
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(dict)):
        embedding_weights[i+1, :] = model_word[dict[i]]
        w2dic[dict[i]] = i+1
        
    print(len(w2dic))
    print(len(embedding_weights))
    np.save(result_dir + 'w2dic.npy', w2dic)
    np.save(result_dir + 'embedding_weights.npy', embedding_weights)
    w2dic={}
    embedding_weights=[]
    
    
    model_word.wv.save_word2vec_format(result_dir + 'Word2Vec.bin' ,binary = True)
    # 保存为另外一种格式，方便查看对应分出来的keys
    model_word.wv.save_word2vec_format(result_dir + 'Word2Vec.txt',binary = False)
    
    w2indx = np.load(result_dir + 'w2dic.npy').item()
    print(len(w2indx))
    print(w2indx['父母'])
    embedding_weights = np.load(result_dir + 'embedding_weights.npy')
    print(len(embedding_weights))
    print(embedding_weights[0])
    print(embedding_weights[2])
    
    """
    # 追加新词    
    model = Word2Vec.load(result_dir + "Word2Vec.pkl")
    model.build_vocab(X_Vec, update = True)
    model.train(X_Vec, total_examples = model.corpus_count, epochs = model.epochs)
     # 保存为另外一种格式，方便查看对应分出来的keys
    model.wv.save_word2vec_format(result_dir + 'Word2Vec.txt',binary = False)
    """

def save_word2vec():
    # 加载词向量模型
    model_word = KeyedVectors.load_word2vec_format(result_dir + 'Word2Vec.bin', binary = True)
    dict = list(model_word.vocab.keys())
    input_dim = len(dict) + 1 #下标0空出来给不够10的字
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(dict)):
        embedding_weights[i+1, :] = model_word[dict[i]]
        w2dic[dict[i]] = i+1
        
    print(len(len(w2dic)))
    print(len(len(embedding_weights)))
    np.save(result_dir + 'w2dic.npy', w2dic)
    np.save(result_dir + 'embedding_weights.npy', embedding_weights)
    w2dic={}
    embedding_weights=[]


# 1、获取文件数据
X_Vec = loadfile()
# 2、将文件数据jieba分词
X_Vec = file_jieba_cut(X_Vec)
# 3、词向量训练，构造词向量字典
word2vec_train(X_Vec)
"""
save_word2vec()
"""
