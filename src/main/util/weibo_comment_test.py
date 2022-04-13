from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import sequence
import jieba
import re
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from spider import newsSpiderDb as db
from config import sltg_config as sltg_config
from self_attention import Self_Attention
import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(sltg_config.logging_path)
logger = logging.getLogger('spider')

version = '2022-04-13-12-13'
root_path = project_path + '/data/dataset/weibo/'
results_dir = project_path + '/result/textCNN/'
file_name = 'weibo'
lstm_input = 128

def file_jieba_cut(text):
    content_result = []
    for document in text:
        content_result.append(jiebacut(clean_str_sst(document)))
    return content_result

# 去除特殊字符，前后空格和全部小写
def clean_str_sst(string):
    string = re.sub('[，。:,.； |-“”""——_+&;@、《》～（）())#O！：【】\ufeff]', "", string)
    return string.strip().lower()

def jiebacut(text):
    # 将语句分词
    ret = []
    sent_list = jieba.cut(text, cut_all = False) #精确模式
    ret = list(sent_list)
    return ret

# 去除停顿词
def data_prepare(text):
    stop_words = stop_words_list()
    content_result = []
    for document in text:
        ret = []
        for word in document:
            if word not in stop_words:
                ret.append(word)
        content_result.append(ret)
    return content_result

# 获取停顿词
def stop_words_list(filepath = sltg_config.stop_words_path):
    stop_words = {}
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        line = line.strip()
        stop_words[line] = 1
    return stop_words

def get_w2dic():
    return np.load(sltg_config.w2dic_path, allow_pickle=True).item()

def data2index(w2indx, text):
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

def transfer_word2vec(text, w2dic):
    # 将文件数据jieba分词
    content_text = file_jieba_cut(text)
    # 对数据进行预处理，去除停顿词
    content_text = data_prepare(content_text)
    # 文本转关键词序列号数组
    content_index = data2index(w2dic, content_text)
    index = sequence.pad_sequences(content_index,  padding='post', maxlen=lstm_input)
    return index
    
def predict(id, content):
    w2dic = get_w2dic()
    content_index = transfer_word2vec(content, w2dic)
    model = load_model(sltg_config.lstm_path, custom_objects = {
        'Self_Attention': Self_Attention})
    # 预测得到结果
    result = model.predict(content_index)
    #输出结果
    content_score = float('{:.2}'.format(result[0][0]))
    
    textCNN_model = load_model(os.path.join(results_dir, version, 'TextCNN.h5'))
    comments = db.query_news_comment(id)
    i = len(comments)
    comment_score = float(0)
    if i != 0 :
        comments_index = transfer_word2vec(comments, w2dic)
        results = textCNN_model.predict(comments_index)
        for result in results:
            comment_score += float('{:.2}'.format(result[0]))
        comment_score = '{:.2}'.format(comment_score / i)
    else : 
        comment_score = None
    
    return str(content_score), comment_score

if __name__ == '__main__':
    real, fake = [], []
    with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            try:
                content_list = []
                content_list.append(content)
                content_result, comment_result = predict(tid, content_list)
                if label == 'non-rumor' or label == 'true' :
                    real.append(content_result + "\t" + comment_result)
                elif label == 'false':
                    fake.append(content_result + "\t" + comment_result)
                else:
                    print(label)
            except Exception :
                logger.error(u'train 评论检测异常:%s', tid) 
            

    with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            try:
                content_list = []
                content_list.append(content)
                content_result, comment_result = predict(tid, content_list)
                if label == 'non-rumor' or label == 'true' :
                    real.append(content_result + "\t" + comment_result)
                elif label == 'false':
                    fake.append(content_result + "\t" + comment_result)
                else:
                    print(label)
            except Exception :
                logger.error(u'dev 评论检测异常:%s', tid) 


    with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            try:
                content_list = []
                content_list.append(content)
                content_result, comment_result = predict(tid, content_list)
                if label == 'non-rumor' or label == 'true' :
                    real.append(content_result + "\t" + comment_result)
                elif label == 'false':
                    fake.append(content_result + "\t" + comment_result)
                else:
                    print(label)
            except Exception :
                logger.error(u'test 评论检测异常:%s', tid)
                
    rW = open(root_path + 'real_test.txt', 'a', encoding='UTF-8')
    rW.write('\n'.join(real))
    rW.close()

    fW = open(root_path + 'fake_test.txt', 'a', encoding='UTF-8')
    fW.write('\n'.join(fake))
    fW.close()