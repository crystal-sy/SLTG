# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 16:47:09 2021

@author: styra
"""
from bs4 import BeautifulSoup as bs #用于数据抽取
import requests
import json
import os
import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
sys.path.append(project_path)

from config import sltg_config as config
from spider import newsSpiderDb as db
from util import common as util
import time
import js2py
import urllib.parse

import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(config.logging_path)
logger = logging.getLogger('spider')

def tencent_fact_process(url, news):
    detail = {}  # 创建一个字典，存放URL、title、newstime等信息
    detail['id'] = news['id']
    detail['title'] = news['title']
    detail['url'] = url  # 将URL时间存入detail字典中的相应键值中
    detail['date'] = news['date']
    detail['result'] = news['explain']
    page = requests.get(url).content  # 使用requests.get方法获取网页代码，由于bs4可以自动解码URL的编码，所以此处不需要decode
    html = bs(page, 'html.parser')  # 使用html解析器
      # 将新闻标题以文本形式存入detail字典中的相应键值中
    artibody = html.find(class_='question text')  # 使用find方法，获取新闻网页中的article信息
    detail['artibody'] = artibody.text
    checkContent = html.find(class_='check_content_bottom') 
    detail['from'] = checkContent.find_all('span', recursive=False)[1].text.replace('来源 :', '')
    detail['original_url'] = checkContent.find('a')['href']
    detail['spider'] = config.tencentfact_spider
    detail['detection_type'] = getDetectionType(news['explain'])
    return detail

def getDetectionType(explain):
    detection_type_list = {
        '谣言' : 0, # 虚假
        '假新闻' : 0, # 虚假
        '伪科学' : 0, # 虚假
        '伪常识' : 0, # 虚假
        '疑似诈骗' : 1, # 疑似诈骗
        '有失实' : 1, # 疑似诈骗
        '确实如此' : 2, # 真实
        '确有此事' : 2, # 真实
        '尚无定论' : 3, # 待定
        '分情况' : 4, # 分情况
        '存在争议' : 4, # 分情况
        '其他' : 5  # 其他
        }
    detection_type = detection_type_list.get(explain)
    if detection_type == None :
        return 5
    else :
        return detection_type
    

def save_content(detail):
    news_dir = config.tencentfact_dir + detail['date'] + os.sep
    dir_is_Exists = os.path.exists(news_dir)
    if not dir_is_Exists:
        os.makedirs(news_dir) 
    file = news_dir + detail['id'] +'.txt'
    fp = open(file, 'w+', encoding = 'UTF-8')
    content = detail['artibody'].replace('\n', '').replace('\r', '')
    if (content == '') :
        content = detail['title']
    fp.write(json.dumps(content, ensure_ascii = False))
    fp.close()
    return file
    
def getTencentToken():
    try:
        timestamp = int(round(time.time() * 1000))
        CryptoJS = js2py.require('crypto-js')
        key = config.tencentfact_key
        sign = config.tencentfact_sign
        ciphertext = CryptoJS.DES.encrypt(str(timestamp) + '-' + sign, key).toString()
        tencentToken = urllib.parse.quote(ciphertext)
        return tencentToken, timestamp
    except Exception:
        logger.error(u'getTencentToken failed.')
        return '', ''

def check_date(news_date, last_date) :
    if last_date != '' and last_date >= news_date:
        return True
    else:
        return False
    
def news_process(news, news_url):
    try:
        #抽取模块使用bs4
        detail = tencent_fact_process(news_url, news)
        #存储模块 保存到txt
        filePath = save_content(detail)
        # 新闻关键词
        detail['news_theme'] = util.getNewsTheme(filePath)
        # 存数据库
        db.insert_news_knowledge(detail)
    except Exception as e:
        logger.error(u'tencent_fact_process url：%s 请求失败', news_url)
        logger.error(e)
    
def main():
    url_filter_list = []
    #使用BeautifulSoup抽取模块和存储模块
    tencentToken, timestamp = getTencentToken()
    page = -1 #设置爬虫初始爬取的页码
    again_time = 0
    go_on = True
    last_date = db.query_news_knowledge_last_date(config.tencentfact_spider)
    while go_on:
        tencentfact_url = config.tencentfact_url.format(tencentToken, page, timestamp)
        data = requests.get(tencentfact_url) #拼接URL，并获取索引页面信息
        if data.status_code == 200:
            wy_content = data.content.decode(encoding = 'UTF-8')
            data_json = json.loads(wy_content.replace('jsonp(', '').replace(')', '').encode())
            code = data_json.get('code')
            if code == 0:
                #将获取的数据json化
                factList = data_json.get('content') #获取result节点下data节点中的数据，此数据为新闻详情页的信息
                if len(factList) == 0 :
                    go_on = False
                    break
                
                #从新闻详情页信息列表news中，使用for循环遍历每一个新闻详情页的信息
                for news in factList:  
                    if check_date(news['date'], last_date):
                        go_on = False
                        break
                    # 查重
                    news_url = config.tencentfact_news_url.format(news['id'])
                    if news_url not in url_filter_list:
                        url_filter_list.append(news_url) #将爬取过的URL放入list中
                        news_process(news, news_url)
                        
                logger.info(u'tencent_fact_process page：%s 页处理完成', page)
                page += 1 #页码自加1
                
            if code == -1 :
                if again_time > 3:
                    logger.error(u'get tencent token again_time over three.')
                    go_on = False
                    break
                
                again_time += 1
                tencentToken, timestamp = getTencentToken()
            
if __name__ == '__main__':
    main()
