# -*- coding: utf-8 -*-
"""
@author: styra
"""
import jieba.analyse
import sys

# 读文件
def readFile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        contentStr = f.read()
    f.close()
    return contentStr


def jiebaTdidf(file):
    tf_idf_words = jieba.analyse.extract_tags(file, topK=3, withWeight=True, 
                                              allowPOS=('n','nr','ns'))
    themes = []
    for theme in tf_idf_words:
        themes.append(theme[0])
    return ','.join(themes)

 
if __name__ == '__main__':
    file = sys.argv[1]
    #file = "E:\\newslist\\TencentFact\\2022-01-14\\dfdcb18184c60cec4d58125922c3de6e.txt"
    print(jiebaTdidf(readFile(file)))
    