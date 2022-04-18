# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 12:04:38 2022

@author: styra
"""
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from config import sltg_config as sltg_config
import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(sltg_config.logging_path)
logger = logging.getLogger('spider')

root_path = project_path + '/data/dataset/weibo/'

if __name__ == '__main__':
    logger.info(u'textCNN检测开始') 
    real = []
    real_comment = []
    fake= []
    fake_comment = []
    
    real_all = []
    real_content = []
    fake_all = []
    fake_content = []
    with open(root_path + "real_test.txt", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            new_id, content_test, comment_test = line.strip().split("\t")
            real_all.append(new_id)
            if float(content_test) < 0.5:
                real_content.append(content_test)
                
            if comment_test != '##' :
                real.append(new_id)
                if float(comment_test) < 0.5:
                    real_comment.append(comment_test)
        input.close()
    real_content_test = len(real_content) / len(real_all)   
    real_content_tp =  len(real_content)
    real_content_fn =  len(real_all)-len(real_content)
    print(len(real_all))  
    print('SLG-tp:',real_content_tp)  
    print('SLG-fn:',real_content_fn)    
    logger.info(u'textCNN真实新闻内容检测：%s', real_content_test)
    
    real_test = len(real_comment) / len(real)      
    print(len(real))        
    logger.info(u'textCNN真实新闻评论检测：%s', real_test) 
    
    with open(root_path + "fake_test.txt", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            new_id, content_test, comment_test = line.strip().split("\t")
            fake_all.append(new_id)
            if float(content_test) >= 0.8:
                fake_content.append(content_test)
                
            if comment_test != '##' :
                fake.append(new_id)
                if float(comment_test) >= 0.5:
                    fake_comment.append(comment_test)
        input.close()
    fake_test = len(fake_content) / len(fake_all)    
    fake_content_tn =  len(fake_content)
    fake_content_fp =  len(real_all)-len(fake_content) 
    print('SLG-fp:',fake_content_fp)
    print('SLG-tn:',fake_content_tn)  
    test_all = real_content_tp+real_content_fn+fake_content_tn+fake_content_fp
    acc = (real_content_tp+fake_content_tn)/test_all
    pre = (real_content_tp)/(real_content_tp+fake_content_fp)
    recall = real_content_tp/(real_content_tp+real_content_fn)
    F1 = 2*pre*recall/(pre+recall)
    print(acc)
    print(pre)
    print(recall)
    print(F1)
    
    print(len(fake_all))        
    logger.info(u'textCNN虚假新闻内容检测：%s', fake_test)
    
    fake_test = len(fake_comment) / len(fake)
    print(len(fake))
    logger.info(u'textCNN虚假新闻评论检测：%s', fake_test) 
    
                
