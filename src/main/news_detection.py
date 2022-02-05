# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:10:59 2021

@author: styra
"""
import numpy as np
import jieba

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
from tensorflow.keras.models import load_model
import time
import re 
from tensorflow.keras.preprocessing import sequence
import sys


voc_dim = 256 # word的向量维度
data_dir = 'D:\\hscode\\data\\'
result_dir = 'D:\\hscode\\result\\'

class analysis():
    def __init__(self, content, comment):
        self.content = content
        self.comment = comment
        
    def loadfile(self, content, comment):
        #文件输入
        content_text = []
        comment_text = []
        with open(content, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                content_text.append(line)
            f.close()
        
        if comment != '':
            with open(comment, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    comment_text.append(line)
                f.close()
        return content_text, comment_text
    
    def file_jieba_cut(self, content_text, comment_text):
        now = int(time.time())
        timeArray = time.localtime(now)
        nowTime = time.strftime("%Y%m%d%H%M%S", timeArray)
        
        content_result = []
        for document in content_text:
            content_result.append(self.jiebacut(self.clean_str_sst(document), nowTime))
            
        comment_result = []
        for document in comment_text:
            comment_result.append(self.jiebacut(self.clean_str_sst(document), nowTime))
        return content_result, comment_result
    
    # 去除特殊字符，前后空格和全部小写
    def clean_str_sst(self, string):
        string = re.sub("[，。 :,.；|-“”——_+&;@、《》～（）())#O！：【】\ufeff]", "", string)
        return string.strip().lower()
    
    def jiebacut(self, text, nowTime):
        # 将语句分词
        ret = [];
        sent_list = jieba.cut(text, cut_all = False) #精确模式
        ret = list(sent_list)
        return ret
    
    # 去除停顿词
    def data_prepare(self, content_text, comment_text):
        stop_words = self.stop_words_list()
        content_result = []
        for document in content_text:
            ret = []
            for word in document:
                if word not in stop_words:
                    ret.append(word)
            content_result.append(ret)
            
        comment_result = []
        for document in comment_text:
            ret = []
            for word in document:
                if word not in stop_words:
                    ret.append(word)
            comment_result.append(ret)
        return content_result, comment_text
    
    # 获取停顿词
    def stop_words_list(self, filepath = data_dir + 'stop_words.txt'):
        stop_words = {}
        for line in open(filepath, 'r', encoding='utf-8').readlines():
            line = line.strip()
            stop_words[line] = 1
        return stop_words
    
    def get_w2dic(self, version):
        if version is None:
            w2dic = np.load(result_dir + 'w2dic.npy', allow_pickle=True).item()
        else :
            w2dic = np.load(result_dir + version + '\\w2dic.npy', allow_pickle=True).item()
        return w2dic
    
    def data2index(self, w2indx, text):
        data = []
        for sentence in text:
            new_txt = []
            for word in sentence:
                try:
                    new_txt.append(w2indx[word])
                except:
                    new_txt.append(0)
            data.append(new_txt)
        return data 

    def transfer_word2vec(self, content, comment, version):
        # 1、获取文件数据
        content_text, comment_text = self.loadfile(content, comment);
        # 2、将文件数据jieba分词
        content_text, comment_text = self.file_jieba_cut(content_text, comment_text)
        # 3、对数据进行预处理，去除停顿词
        content_text, comment_text = self.data_prepare(content_text, comment_text)
        # 4、转换为词向量
        w2dic = self.get_w2dic(version)
        # 5、文本转关键词序列号数组
        index = self.data2index(w2dic, content_text)
        # 6、 序列预处理pad_sequences()序列填充,前面添0到voc_dim长度
        content_index = sequence.pad_sequences(index, maxlen = voc_dim)
        return content_index
    
    def rumor_predict(self, content, comment):
        content = content.replace("\n", "")
        comment = comment.replace("\n", "")
        if content == '' :
            return "请输入新闻内容的文件路径"
        else :
            version = 'lstm_2'
            # 对数据做预处理，获取内容和评论的词向量
            content_index = self.transfer_word2vec(content, comment, version)
            # 加载算法模型
            if version is None:
                model = load_model(result_dir + "lstm.h5")
            else :
                model = load_model(result_dir + version + "\\lstm.h5")
            # 预测得到结果
            result = model.predict(content_index)
            #输出结果
            score = result[0][0]
            return '{:.2%}'.format(score)
        
    def main(self):
        try:
            return self.rumor_predict(self.content, self.comment)
        except Exception as e:
            return e
 
    
if __name__ == '__main__':
    contentFile = sys.argv[1]
    commentFile = sys.argv[2]
    test = analysis(contentFile, commentFile)
    print(test.main())
        




