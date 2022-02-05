# -*- coding: utf-8 -*-
"""
@author: styra
"""
from sklearn.decomposition import LatentDirichletAllocation  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from wordcloud import WordCloud, ImageColorGenerator
from gensim import corpora, models
import matplotlib.pyplot as plt 
from PIL import Image
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import os
import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
stopword_path = 'D:\hscode\data\stop_words.txt' 
picture_path = 'D:\sycode\SLTG\src\main\spider\ii.png'
sys.path.append(project_path)

from config import sltg_config as config
from spider import newsSpiderDb as db

# 
def tf_idf_news():
    
    # 读入数据
    filepath = config.wy_dir +'2022-01-19' + '\\'#文件路径
    for file in os.listdir(filepath):
        
        filedata = fc_theme(readFile(filepath+file))
        newsLda = LDA(filedata[1])
        themeData = jiebaTdidf(readFile(filepath+file))
        print(themeData)    
        #saveNewsTheme(themeData)
        #drawWordCloud(filedata)
      

# 读文件
def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contentStr = f.read()
    f.close()
    return contentStr


def jiebaTdidf(file):
    
    tf_idf_words = jieba.analyse.extract_tags(file, topK=3, withWeight=True, allowPOS=('n','nr','ns'))
    return tf_idf_words

def plt_imshow(x, tx=None, show=True):
    if tx is None:
        fig, tx = plt.subplots()
    tx.imshow(x)
    tx.axis("off")
    if show: plt.show()
    return tx

def drawWordCloud(document):
    # 处理图片
    new_mask = np.array(Image.open(picture_path))
    news_wc_colors = ImageColorGenerator(new_mask)
    
    # 处理文本
    stopwords = set(['消息', '这个'])
    words = ' '.join(document)
    # 绘制词图：
    wc_news = WordCloud(
        max_words=200, #显示最大词数
        mask=new_mask,
        font_path='msyh.ttc',
        width = 100,                  # 设置宽为 400px
        height=50,                 # 设置高为 300px
        background_color='white',    # 设置背景颜色为白色
        stopwords=stopwords,         # 设置禁用词，在生成的词云中不会出现 set 集合中的词
        max_font_size=50,           # 设置最大的字体大小，所有词都不会超过 100px
        min_font_size=5,            # 设置最小的字体大小，所有词都不会超过 10px
    )
    wc_news.generate(words)
    wc_news.to_file('img.jpg')
    
    #news_ex = plt_imshow(wc_news)
   
# 分词并统计词频
def fc_theme(fileDoc):
    # 读入停留词文件：
    stopword = readFile(stopword_path)
    # 构建停留词词典：
    stopword = stopword.split('\n')
    worddict = {}
    wordlist = []
    for w in jieba.cut(fileDoc, cut_all=False):  # cut_all=False为精确模式，=True为全模式
        # print(w)
        if (w not in stopword) and w != '' and w != ' ' and w != None and w != '\n' and len(w) >= 2:
            # print(w + '-')
            wordlist.append(w)
            try:
                worddict[w] = worddict[w] + 1
            except:
                worddict[w] = 1
    return worddict,wordlist#[worddict, wordlist]

def tf_idf(document):#计算TF_IDF值
    doc = ['aaa bbb cd cd cnd and', 'bbb ccc cd', 'aaa fff']
    transformer = TfidfVectorizer()
    tfidf_model = transformer.fit(doc)
    print(tfidf_model.vocabulary_)
    doc_tfidf_id = tfidf_model.get_feature_names_out()# 得到每个特征对应的id值：即上面数组的下标
    sparse_result = tfidf_model.transform(doc)
    weight = sparse_result.toarray()
    # 构建词与tf-idf的字典：
    doc_tfidf = {}
    for i in range(len(weight)):
        for j in range(len(doc_tfidf_id)):
            if doc_tfidf_id[j] not in doc_tfidf:
                doc_tfidf[doc_tfidf_id[j]] = weight[i][j]
            else:
                doc_tfidf[doc_tfidf_id[j]] = max(doc_tfidf[doc_tfidf_id[j]], weight[i][j])
    print(doc_tfidf)
    # 按值排序：
    alldata = []
    idList = sorted(doc_tfidf.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(1, 1000 if len(idList) > 600 else len(idList)):
        print(idList[i][0], idList[i][1])
        alldata.append([idList[i][0], idList[i][1]])
    print(alldata)

def LDA(doc):#计算LDA值
    #doc = ' '.join(doc)

    #构造词典
    dictionary = corpora.Dictionary([doc])
    # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
    corpus = [dictionary.doc2bow(words) for words in [doc]]
    # lda模型，num_topics设置主题的个数
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
    
    maxValue = 0.00
    maxTopic = []
    # 打印所有主题，每个主题显示5个词
    topics = lda.print_topics(num_words=3)
    for e, values in enumerate(lda.inference(corpus)[0]):
        for ee, value in enumerate(values):
            # print('\t主题%d推断值%.2f' % (ee, value))
            if value > maxValue:
                maxValue = value
                maxTopic = topics[ee]
    print(maxTopic[1])
                

def saveNewsTheme(topic):
    return 

 
if __name__ == '__main__':
    '''
    提取关键词
    '''
    tf_idf_news()
    