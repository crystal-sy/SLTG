# -*- coding: utf-8 -*-
"""
@author: styra
"""

from wordcloud import WordCloud
import imageio
import matplotlib.pyplot as plt 
import os
import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
stopword_path = 'D:\hscode\data\stop_words.txt' 

img_path = r'D:\sycode\SLTG-VUE\sltg-ui\src\assets\images'

sys.path.append(project_path)

from spider import newsSpiderDb as db

# 
def tf_idf_news():
    # 读入数据
    drawWordCloud(db.query_news_keyword_per_month_real(True), "real")
    drawWordCloud(db.query_news_keyword_per_month_real(False), "detect")

# 读文件
def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contentStr = f.read()
    f.close()
    return contentStr

def plt_imshow(x, tx=None, show=True):
    if tx is None:
        fig, tx = plt.subplots()
    tx.imshow(x)
    tx.axis("off")
    if show: plt.show()
    return tx

def drawWordCloud(document,type_news):
    # 处理图片
    new_mask = imageio.imread(img_path + os.sep + 'model.png')
    
    # 处理文本
    stopwords = readFile(stopword_path).split('\n')
    #document = document.strip(',')
    words = document
    # 绘制词图：
    wc_news = WordCloud(
        max_words=2000, #显示最大词数
        mask=new_mask,
        font_path='msyh.ttc',
        width = 400,                  # 设置宽为 400px
        height = 300,                 # 设置高为 300px
        background_color='white',    # 设置背景颜色为白色
        stopwords=stopwords,         # 设置禁用词，在生成的词云中不会出现 set 集合中的词
        max_font_size=70,           # 设置最大的字体大小，所有词都不会超过 100px
        min_font_size=5,            # 设置最小的字体大小，所有词都不会超过 10px
    )
    wc_news.generate(words)
    wc_news.to_file(img_path + os.sep + type_news + '.jpg')
    
    #news_ex = plt_imshow(wc_news)

if __name__ == '__main__':
    '''
    提取关键词
    '''
    tf_idf_news()
    