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

"""
keras开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的
高阶应用程序接口,进行深度学习模型的设计、调试、评估、应用和可视化
"""
# Keras Preprocessing是Keras深度学习库的数据预处理和数据增补模块
# sequence进行数据的序列预处理，如：序列填充
from tensorflow.keras.preprocessing import sequence
# keras数据处理工具库
import tensorflow.keras.utils as kerasUtils
# 加载整个模型结构
from tensorflow.keras.models import load_model
""" 
keras.layers是keras的核心网络层
keras的层主要包括：常用层（Core）、卷积层（Convolutional）、池化层（Pooling）、
局部连接层、递归层（Recurrent）、嵌入层（ Embedding）、高级激活层、规范层、
噪声层、包装层，当然也可以编写自己的层。
"""
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer
import tensorflow as tf
import sys
# multiprocessing包是Python中的多进程管理包。 与threading.Thread类似,
# 它可以利用multiprocessing.Process对象来创建一个进程。
import multiprocessing
import time
import os
import re
from sklearn.metrics import accuracy_score, classification_report

# CPU运行
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 设置递归深度
sys.setrecursionlimit(1000000)

# 使得随机数据可预测
np.random.seed()

# 参数配置
cpu_count = multiprocessing.cpu_count() - 4  # 4CPU数量
voc_dim = 128 # word的向量维度
lstm_input = 128 # lstm输入维度
epoch_time = 10 # epoch 100
batch_size = 16 # batch 32
now = int(time.time())
timeArray = time.localtime(now)
nowTime = time.strftime("%Y%m%d%H%M%S", timeArray)
data_dir = 'data/'
result_dir = 'result/lstm_attention/' + nowTime + '/'
model_dir = 'result/lstm_attention/'

class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim 
        })
        return config

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = backend.dot(x, self.kernel[0])
        WK = backend.dot(x, self.kernel[1])
        WV = backend.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",backend.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = backend.batch_dot(WQ, backend.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5) #开根号，归一化系数
        QK = backend.softmax(QK)
        print("QK.shape", QK.shape)

        V = backend.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

def loadfile():
    #文件输入
    neg = []
    pos = []
    with open(data_dir + 'dataset/weibo/real.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            pos.append(line)
        f.close()
    with open(data_dir + 'dataset/weibo/fake.txt', 'r', encoding='UTF-8') as f:
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
        result.append(jiebacut(clean_str_sst(document)) )
    return result

# 去除特殊字符，前后空格和全部小写
def clean_str_sst(string):
    string = re.sub("[，。:,.；|-“”——_+&;@、《》～（）())#O！：【】\ufeff]", "", string)
    return string.strip().lower()

def jiebacut(text):
    # 将语句分词
    ret = []
    sent_list = jieba.cut(text, cut_all = False) #精确模式
    ret = list(sent_list)
    
    # 追加写入
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fW = open(result_dir + 'jieba_result.txt', 'a', encoding='UTF-8')
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
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fW = open(result_dir + 'prepare_result.txt', 'a', encoding='UTF-8')
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
    w2indx = np.load(data_dir + 'word2vec/128/w2dic.npy', allow_pickle=True).item()
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    # 追加写入
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    fW = open(result_dir + 'data2index_result.txt', 'a', encoding='UTF-8')
    fW.write(''.join(str(i) for i in data))
    fW.write('\n')
    fW.close()
    w2indx = []
    return data 

def test_lstm(x_test, y_test, model):
    print("Evaluate...")
    np.random.seed(200)
    np.random.shuffle(x_test) 
    np.random.seed(200)
    np.random.shuffle(y_test)
    y_pred_one_hot = model.predict(x=x_test, batch_size=batch_size)
    y_pred = tf.math.argmax(y_pred_one_hot, axis=1)

    print('\nTest accuracy: {}\n'.format(accuracy_score(y_test, y_pred)))
    print('Classification report:')
    target_names = ['class {:d}'.format(i) for i in np.arange(2)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=2))    

# 1、获取文件数据
X_Vec, y = loadfile()
# 2、将文件数据jieba分词
X_Vec = file_jieba_cut(X_Vec)
# 3、对数据进行预处理，去除停顿词
X_Vec = data_prepare(X_Vec)
# 4、文本转关键词序列号数组
index = data2index(X_Vec)
# 5、 序列预处理pad_sequences()序列填充,前面添0到voc_dim长度
index2 = sequence.pad_sequences(index)

version = '20220321005053'
model = load_model(model_dir + version + '\\lstm.h5', custom_objects = {
    'Self_Attention': Self_Attention})
    
# 7、情感模型测试
test_lstm(index2, y, model)
