# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:10:59 2021

@author: styra
"""
import jieba
import numpy as np 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
from tensorflow.keras.models import load_model
import time
import re 
import sys
from config.sltg_config import stop_words_path, w2dic_path, lstm_path, textCNN_path, random_forest_path
from self_attention import Self_Attention
from tensorflow.keras.preprocessing import sequence
import joblib

lstm_input = 128 # lstm输入维度

class analysis():
    def __init__(self, content, comment, w2dic, model, isFile = True):
        self.content = content
        self.comment = comment
        self.w2dic = w2dic
        self.model = model
        self.isFile = isFile
        
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
        string = re.sub('[，。:,.； |-“”""——_+&;@、《》～（）())#O！：【】\ufeff]', "", string)
        return string.strip().lower()
    
    def jiebacut(self, text, nowTime):
        # 将语句分词
        ret = []
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
    def stop_words_list(self, filepath = stop_words_path):
        stop_words = {}
        for line in open(filepath, 'r', encoding='utf-8').readlines():
            line = line.strip()
            stop_words[line] = 1
        return stop_words
    
    def get_w2dic(self):
        return np.load(w2dic_path, allow_pickle=True).item()
    
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

    def transfer_word2vec(self, content, comment):
        # 1、获取文件数据
        if self.isFile :
            content_text, comment_text = self.loadfile(content, comment)
        else : 
            content_text, comment_text = content, comment
        # 2、将文件数据jieba分词
        content_text, comment_text = self.file_jieba_cut(content_text, comment_text)
        # 3、对数据进行预处理，去除停顿词
        content_text, comment_text = self.data_prepare(content_text, comment_text)
        # 4、转换为词向量
        w2dic = self.w2dic
        if w2dic is None :
            w2dic = self.get_w2dic()
        # 5、文本转关键词序列号数组
        content_index = self.data2index(w2dic, content_text)
        index_content = sequence.pad_sequences(content_index,  padding='post', maxlen=lstm_input)
        
        index_comment = []
        if len(comment_text) > 0 :
            comment_index = self.data2index(w2dic, comment_text)
            index_comment = sequence.pad_sequences(comment_index,  padding='post', maxlen=lstm_input)
        return index_content, index_comment
    
    def rumor_predict(self, content, comment):
        content = content.replace("\n", "")
        comment = comment.replace("\n", "")
        if content == '' :
            return "请输入新闻内容的文件路径"
        else :
            # 对数据做预处理，获取内容和评论的词向量
            content_index, comment_index = self.transfer_word2vec(content, comment)
            # 加载算法模型
            model = self.model
            if model is None:
                model = load_model(lstm_path, custom_objects = {
                    'Self_Attention': Self_Attention})
            # 预测得到结果
            result = model.predict(content_index)
            #输出结果
            content_score = '{:.2}'.format(result[0][0])
            print('新闻内容检测结果:', content_score)
            
            i = len(comment_index)
            comment_score = float(0)
            if i != 0 :
                textCNN_model = load_model(textCNN_path)
                results = textCNN_model.predict(comment_index)
                for result in results:
                    comment_score += float('{:.2}'.format(result[0]))
                comment_score = '{:.2}'.format(comment_score / i)
                print('新闻评论检测结果:', comment_score) 
                
                random_forest_model = joblib.load(random_forest_path)
                score = []
                score.append(content_score)
                score.append(comment_score)
                scores = []
                scores.append(score)
                result = random_forest_model.predict(scores)
                return result[0]
        return content_score
        
    def main(self):
        try:
            return self.rumor_predict(self.content, self.comment)
        except Exception as e:
            return e
 
    
if __name__ == '__main__':
    contentFile = sys.argv[1]
    commentFile = sys.argv[2]
    test = analysis(contentFile, commentFile, None, None)
    print(test.main())
        




