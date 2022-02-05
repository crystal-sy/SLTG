# -*- coding: utf-8 -*-
"""
@author: styra
"""
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation  
import pandas as pd
import numpy as np
import jieba
import os
import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
stopword_path = 'D:\hscode\data\stop_words.txt' 
sys.path.append(project_path)

from config import sltg_config as config
from spider import newsSpiderDb as db

# 
def tf_idf_news():
    
    # 读入数据
    filepath = config.sina_dir +'2021-11-27' + '\\'#文件路径
    for file in os.listdir(filepath):
        filedata = fc_theme(readFile(filepath+file))
        themeData = tf_idf(filedata)
        saveNewsTheme(themeData)
      

# 读文件
def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contentStr = f.read()
    f.close()
    return contentStr

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
    print(worddict)
    return wordlist #[worddict, wordlist]

def tf_idf(doc):#计算TF_IDF值
    transformer = TfidfVectorizer(min_df=0.023)
    tfidf_model = transformer.fit(doc)
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

def saveNewsTheme(topic):
    return 

def LDA(doc):#计算LDA值
    Vectorizer = CountVectorizer()
    lda_model = Vectorizer.fit_transform(doc)
    doc_LDA = lda_model.get_feature_names_out()
   
    # LDA主题模型
    lda = LatentDirichletAllocation(n_components=10,  # 主题个数
                                    max_iter=5,      # EM算法的最大迭代次数
                                    learning_method='online',
                                    learning_offset=20., # 仅仅在算法使用online时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
                                    random_state=0)
    docres = lda.fit_transform(lda_model)
    # 类别所属概率
    LDA_corpus = np.array(docres)
  
    # 打印每个单词的主题的权重值
    topic_matrix = lda.components_
    # 类别id
    id = 0
    # 存储数据
    datalist = []
    for tt_m in topic_matrix:
        # 元组形式
        topic_dict = [(name, tt) for name, tt in zip(doc_LDA, tt_m)]
        topic_dict = sorted(topic_dict, key=lambda x: x[1], reverse=True)
        # 打印权重值大于0.6的主题词：
        # tt_dict = [tt_threshold for tt_threshold in tt_dict if tt_threshold[1] > 0.6]
        # 打印每个类别的前20个主题词：
        topic_dict = topic_dict[:20]
        print('主题%d:' % id, topic_dict)
        # 存储：
        datalist += [[topic_dict[i][0], topic_dict[i][1], id]for i in range(len(topic_dict))]
        id += 1
    # 存入excel:
    # df = pd.DataFrame(datalist, columns=['特征词', '权重', '类别'])
    # df.to_excel('./data/LDA_主题分布3.xlsx', index=False)
 
 
if __name__ == '__main__':
    '''
    提取关键词
    '''
    tf_idf_news()