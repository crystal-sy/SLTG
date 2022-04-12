# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:53:29 2021

@author: ~styra~
新闻文本LSTM+Self_Attention算法
"""

"""
numpy是python扩展程序库，支持大量的维度数组与矩阵运算，
此外也针对数组运算提供大量的数学函数库
"""
import numpy as np 
# jieba分词
import jieba
# 自然语言处理NLP神器--gensim，词向量Word2Vec
from matplotlib import pyplot as plt

"""
keras开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的
高阶应用程序接口,进行深度学习模型的设计、调试、评估、应用和可视化
"""
# Keras Preprocessing是Keras深度学习库的数据预处理和数据增补模块
# sequence进行数据的序列预处理，如：序列填充
from tensorflow.keras.preprocessing import sequence
# keras数据处理工具库
import tensorflow.keras.utils as kerasUtils
# 在Keras中有两类主要的模型：Sequential顺序模型和使用函数式API的Model类模型
from tensorflow.keras.models import Sequential
# 加载整个模型结构
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
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
import time
import os
import re
from self_attention import Self_Attention
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# CPU运行
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 设置递归深度
sys.setrecursionlimit(1000000)

# 使得随机数据可预测
np.random.seed()

# 参数配置
voc_dim = 128 # word的向量维度
lstm_input = 128 # lstm输入维度
epoch_time = 20 # epoch 100
batch_size = 32 # batch 32
version = None
now = int(time.time())
timeArray = time.localtime(now)
nowTime = time.strftime("%Y%m%d%H%M%S", timeArray)
data_dir = 'data/'
result_dir = 'result' + os.sep
model_dir = result_dir + 'lstm_attention' + os.sep 
sl_result_dir = model_dir + nowTime + os.sep


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
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neg), dtype=int))) #拼接全1和全0数组
    return X_Vec, y

def file_jieba_cut(text):
    result=[]
    for document in text:
        result.append(jiebacut(clean_str_sst(document)))
    return result

# 去除特殊字符，前后空格和全部小写
def clean_str_sst(string):
    string = re.sub('[，。:,.； |-“”""——_+&;@、《》～（）())#O！：【】\ufeff]', "", string)
    return string.strip().lower()

def jiebacut(text):
    # 将语句分词
    ret = []
    sent_list = jieba.cut(text, cut_all = False) #精确模式
    ret = list(sent_list)
    
    # 追加写入
    if not os.path.exists(sl_result_dir):
        os.mkdir(sl_result_dir)
    fW = open(sl_result_dir + 'jieba_result.txt', 'a', encoding='UTF-8')
    fW.write(' '.join(ret))
    fW.write('\n')
    fW.close()
    return ret

# 去除停顿词
def data_prepare(text):
    stop_words = stop_words_list()
    result = []
    for document in text:
        ret = []
        for word in document:
            if word not in stop_words:
                ret.append(word)
        result.append(ret)
        
    # 追加写入
    if not os.path.exists(sl_result_dir):
        os.mkdir(sl_result_dir)
    fW = open(sl_result_dir + 'prepare_result.txt', 'a', encoding='UTF-8')
    fW.write(''.join(str(i) for i in result))
    fW.write('\n')
    fW.close()
    return result

# 获取停顿词
def stop_words_list(filepath = data_dir + 'stop_words.txt'):
    stop_words = {}
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        line = line.strip()
        stop_words[line] = 1
    return stop_words

def data2index(X_Vec):
    data = []
    word_dict = np.load(result_dir + 'w2dic.npy', allow_pickle=True).item()
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_dict[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
        
    vocab_size = len(word_dict)
    print("Vocabulary size: {:d}".format(vocab_size))
    return data

def train_lstm(embedding_weights, x_train, y_train, x_test, y_test, y_test_1, version):
    if version is None :
        #Keras有两种不同的构建模型-顺序模型
        model = Sequential()
        # 嵌入层
        model.add(Embedding(input_dim=len(embedding_weights),
                           output_dim=voc_dim,
                           mask_zero=True,
                           weights=[embedding_weights],
                           input_length=lstm_input))
        model.add(Self_Attention(128))
        model.add(LSTM(128, activation='softsign')) # 激活函数softsign
        # model.add(Flatten(data_format='channels_last'))
        model.add(Dropout(0.5)) # 防止过拟合
        model.add(Dense(2, kernel_regularizer=regularizers.l2(0.003))) # 全连接层
        # model.add(Dense(2)) # 全连接层
        model.add(Activation('sigmoid'))

        embedding_weights = []
        print ('Compiling the Model...')
        # 均方误差mean_squared_error/mse
        # 平均绝对误差mean_absolute_error/mae
        # 平均绝对百分比误差mean_absolute_percentage_error/mape
        # 均方对数误差mean_squared_logarithmic_error/msle
        model.compile(loss='binary_crossentropy',#hinge  # 损失函数，对数损失
                      optimizer='adam', metrics=['mse', 'acc'])
        #model.compile(loss='binary_crossentropy',#hinge  # 损失函数
                   #   optimizer='adam', metrics=['mae', 'acc'])
    else :
        # 基于之前生成的模型继续训练
        model = load_model(model_dir + version + '\\lstm.h5', custom_objects = {
            'Self_Attention': Self_Attention})

    """
    在 fit 和 evaluate 中 都有 verbose 这个参数
    fit 中的 verbose
    verbose：该参数的值控制日志显示的方式
    verbose = 0    不在标准输出流输出日志信息
    verbose = 1    输出进度条记录
    verbose = 2    每个epoch输出一行记录
    注意： 默认为 1
    
    evaluate 中的 verbose
    verbose：控制日志显示的方式
    verbose = 0  不在标准输出流输出日志信息
    verbose = 1  输出进度条记录
    注意： 只能取 0 和 1；默认为 1
    """
    print ("Train...")  # batch_size=32
    #数据太集中，打乱顺序
    # np.random.seed(200)
    # np.random.shuffle(x_train) 
    # np.random.seed(200)
    # np.random.shuffle(y_train)
 
    h = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_time, 
              verbose=1,
              validation_split = 0.2)
    
    plt.plot(h.history["loss"],label="train_loss")
    plt.plot(h.history["val_loss"],label="val_loss")
    plt.plot(h.history["acc"],label="train_acc")
    plt.plot(h.history["val_acc"],label="val_acc")
    plt.legend()
    plt.savefig(sl_result_dir + 'result.png') # show之前保存图片，之后保存图片为空白
    plt.show()

    print("Evaluate...")
    # np.random.seed(200)
    # np.random.shuffle(x_test) 
    # np.random.seed(200)
    # np.random.shuffle(y_test)
    # np.random.shuffle(y_test_1)
    score = model.evaluate(x_test, y_test_1, batch_size=batch_size)
    print ('Test score:', score)
    
    # 保存结果
    if not os.path.exists(sl_result_dir):
        os.mkdir(sl_result_dir)
    yaml_string = model.to_yaml()
    with open(sl_result_dir + 'lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save(sl_result_dir + 'lstm.h5')
    # kerasUtils.plot_model(model, to_file = result_dir + 'model.png')
    
    # 展开模型参数
    loadModel = load_model(sl_result_dir + 'lstm.h5', custom_objects = {
        'Self_Attention': Self_Attention})
    with open(sl_result_dir + 'modelsummary.txt', 'w') as f:
        loadModel.summary(print_fn=lambda x: f.write(x + '\n'))
        
    y_pred_one_hot = model.predict(x=x_test, batch_size=batch_size)
    y_pred = tf.math.argmax(y_pred_one_hot, axis=1)

    print('\nTest accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
    print('Classification report:')
    target_names = ['class {:d}'.format(i) for i in np.arange(2)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# 1、获取文件数据
X_Vec, y = loadfile()
# 2、将文件数据jieba分词
X_Vec = file_jieba_cut(X_Vec)
# 3、对数据进行预处理，去除停顿词
X_Vec = data_prepare(X_Vec)
# 4、文本转关键词序列号数组
index = data2index(X_Vec)
# 5、 序列预处理pad_sequences()序列填充,尾部添0到voc_dim长度
index2 = sequence.pad_sequences(index,  padding='post', maxlen=lstm_input)
# 6、函数划分训练、测试数据
x_train, x_test, y_train, y_test = train_test_split(index2, y, test_size=0.2)
# 7、将原向量变为one-hot编码，数据转为num_classes数组
y_train = kerasUtils.to_categorical(y_train, num_classes=2)
y_test_1 = kerasUtils.to_categorical(y_test, num_classes=2)

if version is None:
    # 8、获取权重
    embedding_weights = np.load(result_dir + 'embedding_weights.npy', allow_pickle=True)
else :
    embedding_weights = []
# 8、lstm+Self_Attention情感训练
train_lstm(embedding_weights, x_train, y_train, x_test, y_test, y_test_1, version)
