# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:15:49 2021

@author: styra
"""

import pickle
import gensim
import time
import numpy as np
from gensim.models.word2vec import Word2Vec


result_dir = 'D:\\hscode\\result\\'

#content = pickle.load(open(r'D:\\paper\\EANN-KDD18-master\\data\\weibo\\w2v.pickle','rb'), encoding='bytes')
"""
content = gensim.models.KeyedVectors.load_word2vec_format('D:\\paper\\w2v\\news12g_bdbk20g_nov90g_dim128\\news12g_bdbk20g_nov90g_dim128.bin', binary = True)
keys = list(content.wv.vocab.keys())
print(len(keys))

content = pickle.load(open(r'D:\\paper\\EANN-KDD18-master\\data\\weibo\\word_embedding.pickle','rb'), encoding='bytes')
print(content)

"""
"""
t1=time.time()
content = pickle.load(open(result_dir + 'Word2Vec.pkl','rb'), encoding='bytes')
t2=time.time()
print(f"Time took to read pkl: {t2-t1} seconds.")
print(content.wv['父母'])

keys = list(content.wv.vocab.keys())
for key in keys:
    if key == "父母" :
        print(content.wv[key])
"""

"""
t1=time.time()        
content = gensim.models.KeyedVectors.load_word2vec_format('D:\\Upan\\120G\\news12g_bdbk20g_nov90g_dim128\\news12g_bdbk20g_nov90g_dim128.bin', binary = True)
t2=time.time()
print(f"Time took to read 64 bin: {t2-t1} seconds.")
print(content)

keys = list(content.vocab.keys())
for key in keys:
    if key == "父母" :
        print(content[key])
"""

"""        
t1=time.time()
content = gensim.models.KeyedVectors.load_word2vec_format('D:\\Upan\\120G\\news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary = True)
t2=time.time()
print(f"Time took to read 128 bin: {t2-t1} seconds.")
print(content['父母'])
keys = list(content.vocab.keys())
for key in keys:
    if key == "父母" :
        print(content[key])
"""

""" 
t1=time.time()        
content = np.load('D:\\Upan\\120G\\news12g_bdbk20g_nov90g_dim128\\news12g_bdbk20g_nov90g_dim128.model.syn0.npy')
t2=time.time()
print(f"Time took to read 64 npy: {t2-t1} seconds.")

keys = list(content.vocab.keys())
for key in keys:
    if key == "父母" :
        print(content[key])
"""
    
"""    
t1=time.time()
content = np.load('D:\\Upan\\120G\\news_12g_baidubaike_20g_novel_90g_embedding_64.model.syn0.npy')
t2=time.time()
print(f"Time took to read 128 npy: {t2-t1} seconds.")
print(content)

keys = list(content.vocab.keys())
for key in keys:
    if key == "父母" :
        print(content[key])

"""
"""
t1=time.time()        
content = gensim.models.KeyedVectors.load_word2vec_format('D:\\paper\\w2v\\baike_26g_news_13g_novel_229g.bin', binary = True)
t2=time.time()
print(f"Time took to read 64 bin: {t2-t1} seconds.")
print(content)

keys = list(content.vocab.keys())
print(len(keys))
"""

"""
t1=time.time()        
content = np.load('D:\\paper\\w2v\\baike_26g_news_13g_novel_229g.model.wv.vectors.npy')
t2=time.time()
print(f"Time took to read 64 bin: {t2-t1} seconds.")
print(content)

keys = list(content.vocab.keys())
print(len(keys))
"""


content = gensim.models.KeyedVectors.load_word2vec_format('D:\\hscode\\result\\Word2Vec.bin', binary=True)
keys = list(content.vocab.keys())
print(keys)