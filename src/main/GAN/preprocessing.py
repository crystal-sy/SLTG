import re
import numpy as np
from gan_config import dataset_path, SEQ_LENGTH, real_file, generated_num, vocab_file
from tensorflow.keras.preprocessing import sequence
import pandas as pd
from tensorflow.keras import preprocessing
import json
import jieba
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

def loadfile():
    #文件输入
    data = []
    with open(dataset_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data.append(line)
        f.close()
    return data

def file_jieba_cut(text):
    result=[]
    for document in text:
        result.append(jiebacut(clean_str_sst(document)))
    return result

# 去除特殊字符，前后空格和全部小写
def clean_str_sst(str):
    str = re.sub("[，。:,.；|-“”——_+&;@、《》～（）())#O！：[]【】\ufeff]", "", str)
    return str.strip().lower()

def jiebacut(text):
    # 将语句分词
    sent_list = jieba.cut(text, cut_all = False) #精确模式
    return list(sent_list)

def data2index(x_text):
    text_preprocessor = preprocessing.text.Tokenizer(oov_token="<UNK>")
    text_preprocessor.fit_on_texts(x_text)
    x = text_preprocessor.texts_to_sequences(x_text)
    word_dict = text_preprocessor.word_index
    json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
    vocab_size = len(word_dict)
    print("Vocabulary size: {:d}".format(vocab_size))
    
    x = preprocessing.sequence.pad_sequences(x, padding='post', maxlen=SEQ_LENGTH)#将序列转化为经过填充以后得到的一个长度相同新的序列
    return x 


if __name__ == "__main__":
    # 1、获取文件数据
    x_text = loadfile()
    # 2、将文件数据jieba分词
    X_Vec = file_jieba_cut(x_text)
    # 3、文本转关键词序列号数组
    x = data2index(X_Vec)
    np.savetxt(real_file, x[:generated_num], fmt='%i')
