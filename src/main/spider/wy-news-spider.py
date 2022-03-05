# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 20:42:36 2021

@author: styra
"""
from bs4 import BeautifulSoup as bs #用于数据抽取
import requests
import json
import os
import time
import sys
import numpy as np
from tensorflow.keras.models import load_model
from config import sltg_config as config
from spider import newsSpiderDb as db
from util import common as util
import news_detection as detection

import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(config.logging_path)
logger = logging.getLogger('spider')

def news_content_process(news, url, wy_newsDate):
    detail = {}  # 创建一个字典，存放URL、title、newstime等信息
    wy_id = url.split('/')[-1].split('.')[0]
    detail['id'] = wy_id
    detail['url'] = url  # 将URL时间存入detail字典中的相应键值中
    page = requests.get(url).content  # 使用requests.get方法获取网页代码，由于bs4可以自动解码URL的编码，所以此处不需要decode
    html = bs(page, 'html.parser')  # 使用html解析器
    wy_title = html.find(class_='post_title')  # 获取新闻网页中的title信息
    detail['title'] = wy_title.text  # 将新闻标题以文本形式存入detail字典中的相应键值中
    news_content = html.find(class_='post_body')  # 使用find方法，获取新闻网页中的article信息
    detail['artibody'] = news_content.text
    detail['date'] = wy_newsDate
    detail['spider'] = config.wy_spider
    source = news.get('source')
    if source is None :
        detail['from'] = html.find(class_='post_info').find('a').text
    else :
        detail['from'] = source
    return detail


def save_content(detail):
    news_dir = config.wy_dir + detail['date'] + os.sep
    dir_is_Exists = os.path.exists(news_dir)
    if not dir_is_Exists:
        os.makedirs(news_dir) 
    file = news_dir + detail['id']  +'.txt'
    fp = open(file, 'w+', encoding = 'UTF-8')
    content = detail['artibody'].replace('\n', '').replace('\r', '')
    fp.write(json.dumps(content, ensure_ascii = False))
    fp.close()
    return file
    
def news_process(wy_news, wy_news_url, wy_newsDate, w2dic, model):
    try:
        wy_content = news_content_process(wy_news, wy_news_url, wy_newsDate)
        #存储模块 保存到txt
        filePath = save_content(wy_content)
        # 新闻关键词
        wy_content['news_theme'] = util.getNewsTheme(filePath)
        # 新闻检测
        rumor_predict = detection.analysis(filePath, '', w2dic, model)
        wy_content['detection_percent'] = rumor_predict.main()
        if len(wy_content['detection_percent']) > 10:
            print(wy_content['detection_percent'])
            wy_content['detection_percent'] = ''
            
        # 存数据库
        db.insert_news_info(wy_content)
    except Exception as e:
        logger.error(u'wy_news_process url: %s bad request.', wy_news_url)
        logger.error(e)        
    
def get_date(news_time, key):
    try:
        to_format = '%m/%d/%Y %H:%M:%S'
        if key == 'society' :
            to_format = '%a %b %d %H:%M:%S CST %Y'
            
        newsDateArray = time.strptime(news_time, to_format)
        return time.strftime('%Y-%m-%d', newsDateArray)
    except Exception as e:
        logger.error(u'wy_news_time：%s convert failed. ', news_time)
        logger.error(e)

def main(sinceDate):
    url_filter_list = []
    w2dic = np.load(config.w2dic_path, allow_pickle=True).item() 
    model = load_model(config.lstm_path)
    #以API为index开始获取url列表
    for key in config.wy_url_list.keys():
        page = 1
        go_on = True
        while go_on :
            tail = ''
            if page != 1:
                tail = '_0' + str(page)
            data = requests.get(config.wy_url_list[key][0].format(tail)) #拼接URL，并获取索引页面信息
            if data.status_code == 200:  #当请求页面返回200（代表正确）时，获取网页数据
            #将获取的数据json化
                wy_content = data.content.decode(encoding='UTF-8', errors='ignore')
                data_jsons = json.loads(wy_content.replace('data_callback(', '').replace(')', '').encode())
                if len(data_jsons) == 0 :
                    go_on = False
                    break
                
                for wy_news in data_jsons:
                    wy_newsDate = get_date(wy_news['time'], key)
                    if wy_newsDate < sinceDate:
                        go_on = False
                        break

                    # 查重
                    wy_news_url = wy_news['docurl'] 
                    if wy_news_url not in url_filter_list:
                        url_filter_list.append(wy_news_url) #将爬取过的URL放入list中
                        news_process(wy_news, wy_news_url, wy_newsDate, w2dic, model)
                        
                logger.info(u'wy_process key: %s, page：%s 页处理完成', key, page)
                page += 1
            
            if data.status_code != 200:
                logger.error(u'wy_process key: %s, page：%s 页处理失败', key, page)
                go_on = False
                break
        
if __name__ == '__main__':
    sinceDate = sys.argv[1]
    # sinceDate = '2022-03-06'
    main(sinceDate)