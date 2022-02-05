# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:30:33 2022

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


def jiebaTdidf(content):
    tf_idf_words = jieba.analyse.extract_tags(content, topK=3, withWeight=True, 
                                              allowPOS=('n','nr','ns'))
    themes = []
    for theme in tf_idf_words:
        themes.append(theme[0])
    return ','.join(themes)

def getNewsTheme(file):
    return jiebaTdidf(readFile(file))

def getNewsThemeWithContent(content):
    return jiebaTdidf(content)

 
if __name__ == '__main__':
    file = sys.argv[1]
    print(getNewsTheme(file))