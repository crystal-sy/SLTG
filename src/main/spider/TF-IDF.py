# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 19:47:45 2022

@author: styra
"""
 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
sys.path.append(project_path)

from config import sltg_config as config
from spider import newsSpiderDb as db

def tf_idf_news():
    
    # 读入数据
    filepath = config.sina_dir +'2021-11-27' + '\\'#文件路径
    for file in os.listdir(filepath):
        filedata = open(file)   #将文件中数据
        tf_idf(filedata)
        filedata.close()
        
def tf_idf(document):
    
    # min_df: 当构建词汇表时，严格忽略低于给出阈值的文档频率的词条，语料指定的停用词。如果是浮点值，该参数代表文档的比例，整型绝对计数值，如果词汇表不为None，此参数被忽略。
    tfidf_model = TfidfVectorizer(min_df=0.023).fit(document)
    # 得到语料库所有不重复的词
    feature = tfidf_model.get_feature_names()
    # print(feature)
    # print(len(feature))
    # ['一切', '一条', '便是', '全宇宙', '天狗', '日来', '星球']
    # 得到每个特征对应的id值：即上面数组的下标
    # print(tfidf_model.vocabulary_)
    # {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}
 
    # 每一行中的指定特征的tf-idf值：
    sparse_result = tfidf_model.transform(document)
    # print(sparse_result)
 
    # 每一个语料中包含的各个特征值的tf-idf值：
    # 每一行代表一个预料，每一列代表这一行代表的语料中包含这个词的tf-idf值，不包含则为空
    weight = sparse_result.toarray()
 
    # 构建词与tf-idf的字典：
    feature_TFIDF = {}
    for i in range(len(weight)):
        for j in range(len(feature)):
            # print(feature[j], weight[i][j])
            if feature[j] not in feature_TFIDF:
                feature_TFIDF[feature[j]] = weight[i][j]
            else:
                feature_TFIDF[feature[j]] = max(feature_TFIDF[feature[j]], weight[i][j])
    # print(feature_TFIDF)
    # 按值排序：
    print('TF-IDF 排名前十的：')
    alldata = []
   
  
 
def drawWordCloud():
    from wordcloud import WordCloud
    from scipy.misc import imread
    # 读入数据
    filepath = './data/text_2.xlsx'
    data = pd.read_excel(filepath)
    document = list(data['文章内容分词'])
    # 整理文本：
    # words = '一切 一条 便是 全宇宙 天狗 日来 星球' # 样例
    words = ''.join(document)
    # print(words)
    # 设置背景图片：
    b_mask = imread('./image/ciyun.webp')
    # 绘制词图：
    wc = WordCloud(
        background_color="white", #背景颜色
        max_words=2000, #显示最大词数
        font_path="./image/simkai.ttf",  #使用字体
        # min_font_size=5,
        # max_font_size=80,
        # width=400,  #图幅宽度
        mask=b_mask
    )
    wc.generate(words)
    # 准备一个写入的背景图片
    wc.to_file("./image/beijing_2.jpg")
 
if __name__ == '__main__':
    '''
    提取关键词
    '''
    tf_idf_news()
 
    '''
    绘制词云图片
    '''
  # drawWordCloud()
