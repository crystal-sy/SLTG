# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:53:29 2021

@author: ~styra~
新闻文本LSTM算法
"""

"""
numpy是python扩展程序库，支持大量的维度数组与矩阵运算，
此外也针对数组运算提供大量的数学函数库
"""
import numpy as np
# jieba分词
import jieba

from gensim.models.word2vec import Word2Vec
"""
keras开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的
高阶应用程序接口,进行深度学习模型的设计、调试、评估、应用和可视化
"""
# Keras Preprocessing是Keras深度学习库的数据预处理和数据增补模块
# sequence进行数据的序列预处理，如：序列填充
from tensorflow.keras.preprocessing import sequence
# keras数据处理工具库
import tensorflow.keras.utils as utils
# 在Keras中有两类主要的模型：Sequential顺序模型和使用函数式API的Model类模型
from tensorflow.keras.models import Sequential
# 加载整个模型结构
from tensorflow.keras.models import load_model
"""
keras.layers是keras的核心网络层
keras的层主要包括：常用层（Core）、卷积层（Convolutional）、池化层（Pooling）、
局部连接层、递归层（Recurrent）、嵌入层（ Embedding）、高级激活层、规范层、
噪声层、包装层，当然也可以编写自己的层。
"""
# 嵌入层只能作为模型第一层
# mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，
# 该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持
# masking，否则会抛出异常
from tensorflow.keras.layers import Embedding
# 递归层（循环层）包含三种模型：LSTM、GRU和SimpleRNN
from tensorflow.keras.layers import LSTM
# Dense层(全连接层）
# 为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）
# 随机断开输入神经元，Dropout层用于防止过拟合。
# Activation层（激活层对一个层的输出施加激活函数）
from tensorflow.keras.layers import Dense, Dropout, Activation
"""
scikit-learn 是基于 Python 语言的机器学习工具
简单高效的数据挖掘和数据分析工具
可供大家在各种环境中重复使用
"""
# model_selection这个模块主要是对数据的分割，以及与数据划分相关的功能
from sklearn.model_selection import train_test_split

import yaml
import sys
# multiprocessing包是Python中的多进程管理包。 与threading.Thread类似,
# 它可以利用multiprocessing.Process对象来创建一个进程。
import multiprocessing
import time
import os

# CPU运行
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 设置递归深度
sys.setrecursionlimit(1000000)

# 使得随机数据可预测
np.random.seed()

# 参数配置
cpu_count = multiprocessing.cpu_count() - 4  # 4CPU数量
voc_dim = 150 # word的向量维度
min_out = 10 #单词出现次数
window_size = 7 #WordVec中的滑动窗口大小

lstm_input = 150#lstm输入维度
epoch_time = 10#epoch
batch_size = 16 #batch
dir_dev = 'data/'


def loadfile():
    #文件输入
    neg = []
    pos = []
    with open(dir_dev + 'pos_test.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            pos.append(line)
        f.close()
    with open(dir_dev + 'neg_test.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            neg.append(line)
        f.close()
    X_Vec = np.concatenate((pos, neg)) #数组拼接
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neg), dtype=int))) #拼接全1和全0数组
    return X_Vec, y

def one_seq(text):
    now = int(time.time())
    timeArray = time.localtime(now)
    nowTime = time.strftime("%Y%m%d%H%M%S", timeArray)
    
    result=[]
    for document in text:
        result.append(jiebacut(document.replace('\n', ''), nowTime) )
    return result

def jiebacut(text, nowTime):
    # 将语句分词
    ret = []
    sent_list = jieba.cut(text, cut_all = False)
    ret = list(sent_list)
    
    # 追加写入
    fW = open(dir_dev + 'train/result_' + nowTime + '.txt', 'a', encoding='UTF-8')
    fW.write(' '.join(ret))
    fW.write('\n')
    fW.close()
    return ret

def word2vec_train(X_Vec):
    #vector_size:是指特征向量的维度,默认为100.大的size需要更多的训练数据,但是效果会更好.推荐值为几十到几百
    #min_count:可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    #window:表示当前词与预测词在一个句子中的最大距离是多少
    #workers:参数控制训练的并行数。
    #epochs:迭代次数，默认为5
    model_word = Word2Vec(vector_size=voc_dim,
                     min_count=min_out,
                     window=window_size,
                     workers=cpu_count,
                     epochs=5)
    model_word.build_vocab(X_Vec)
    model_word.train(X_Vec, total_examples=model_word.corpus_count, epochs=model_word.epochs)
    model_word.save(dir_dev + 'train/Word2Vec.pkl')

    keys = list(model_word.wv.vocab.keys())
    input_dim = len(keys) + 1 #下标0空出来给不够10的字
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(keys)):
        embedding_weights[i+1, :] = model_word.wv[keys[i]]
        w2dic[keys[i]] = i+1
    return input_dim,embedding_weights,w2dic

def data2index(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    return data 

def train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test):
    #Keras有两种不同的构建模型-顺序模型
    model = Sequential()
    model.add(Embedding(output_dim=voc_dim,
                        input_dim=input_dim,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=lstm_input)) 
    model.add(LSTM(128, activation='softsign')) # 激活函数softsign
    model.add(Dropout(0.5)) #全连接层
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',#hinge  # 损失函数
                  optimizer='adam', metrics=['mean_absolute_error', 'acc'])
    #model.compile(loss='binary_crossentropy',#hinge  # 损失函数
               #   optimizer='adam', metrics=['mae', 'acc'])

    print ("Train...")  # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, verbose=1)

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open(dir_dev + 'train/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save(dir_dev + 'train/lstm.h5')
    print ('Test score:', score)
    loadModel = load_model(dir_dev + 'train/lstm.h5')
    loadModel.summary()


# 1、获取文件数据
X_Vec, y = loadfile()
# 2、将文件数据jieba分词
X_Vec = one_seq(X_Vec)
# 3、词向量训练，构造词向量字典
input_dim,embedding_weights,w2dic = word2vec_train(X_Vec)

index = data2index(w2dic,X_Vec)
# 序列预处理pad_sequences()序列填充
index2 = sequence.pad_sequences(index, maxlen=voc_dim)

# 函数划分训练、测试数据
x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
# 将数据转为num_classes数组
y_train = utils.to_categorical(y_train, num_classes=2)
y_test = utils.to_categorical(y_test, num_classes=2)
# lstm情感训练
train_lstm(input_dim, embedding_weights, x_train, y_train, x_test, y_test)