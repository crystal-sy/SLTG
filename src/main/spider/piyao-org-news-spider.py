# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 16:47:09 2021

@author: styra
"""
from bs4 import BeautifulSoup as bs #用于数据抽取
import requests
import json
import os
import time
import sys

# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
sys.path.append(project_path)

from config import sltg_config as config
from spider import newsSpiderDb as db

import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(config.logging_path)
logger = logging.getLogger('spider')

def news_content_process(news, newsDate):
    detail = {}  # 创建一个字典，存放URL、title、newstime等信息
    detail['id'] = news['DocID']
    detail['url'] = news['LinkUrl']  # 将URL时间存入detail字典中的相应键值中
    detail['title'] = news['Title']
    page = requests.get(news['LinkUrl']).content  # 使用requests.get方法获取网页代码，由于bs4可以自动解码URL的编码，所以此处不需要decode
    html = bs(page, 'html.parser')  # 使用html解析器
    news_content = html.find(class_='con_txt')  # 使用find方法，获取新闻网页中的article信息
    detail['artibody'] = news_content.text
    detail['from'] = news['SourceName']
    # 由于不同的新闻详情页之间使用了不同的标签元素，直接抽取可能会报错，所以此处使用判断语句来进行区分爬取
    detail['date'] = newsDate
    detail['spider'] = config.piyao_org_spider
    detail['news_type'] = 15
    return detail

def save_content(detail):
    news_dir = config.piyao_org_dir + detail['date'] + os.sep
    dir_is_Exists = os.path.exists(news_dir)
    if not dir_is_Exists:
        os.makedirs(news_dir) 
    fp = open(news_dir + str(detail['id']) +'.txt', 'w+', encoding='UTF-8')
    content = detail['artibody'].replace('\n', '').replace('\r', '')
    fp.write(json.dumps(content, ensure_ascii=False))
    fp.close()
    
def news_process(news, newsDate):
    try:
        piyao_org_content = news_content_process(news, newsDate)
        # 存数据库
        db.insert_news_info(piyao_org_content)
        #存储模块 保存到txt
        save_content(piyao_org_content)
    except Exception as e:
        logger.error(u'piyao_org_process url: %s 请求失败', news['LinkUrl'])
        logger.error(e)

def main(sinceDate):
    url_filter_list = []
    page = 1 #设置爬虫初始爬取的页码
    #使用BeautifulSoup抽取模块和存储模块
    #设置爬取页面的上限，
    go_on = True
    while go_on :
        #以API为index开始获取url列表
        data = requests.get(config.piyao_org_url.format(page)) #拼接URL，并获取索引页面信息
        if data.status_code == 200:  #当请求页面返回200（代表正确）时，获取网页数据
            #将获取的数据json化
            data_json = json.loads(data.content)
            newsList = data_json.get('data').get('list') #获取result节点下data节点中的数据，此数据为新闻详情页的信息
            #从新闻详情页信息列表news中，使用for循环遍历每一个新闻详情页的信息
            for news in newsList:
                newsDateArray = time.strptime(news['PubTime'], '%Y-%m-%d %H:%M:%S')
                newsDate = time.strftime('%Y-%m-%d', newsDateArray)
                if newsDate < sinceDate:
                    go_on = False
                    break
                elif newsDate > sinceDate:
                    continue
                
                # 查重
                news_url = news['LinkUrl']
                if news_url not in url_filter_list:
                    url_filter_list.append(news_url) #将爬取过的URL放入list中
                    news_process(news, newsDate)
            page+=1 #页码自加1
        
if __name__ == '__main__':
    sinceDate = sys.argv[1]
    main(sinceDate)