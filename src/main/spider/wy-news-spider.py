# -*- coding: utf-8 -*-
"""
Created on Mon Nov 1 20:42:36 2021

@author: styra
"""
from pybloom_live import ScalableBloomFilter # 用于URL去重的
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

def news_content_process(news, url, wy_newsDate):
    detail = {}  # 创建一个字典，存放URL、title、newstime等信息
    wy_id = url.split('/')[-1].split('.')[0]
    detail['id'] = wy_id
    detail["url"] = url  # 将URL时间存入detail字典中的相应键值中
    page = requests.get(url).content  # 使用requests.get方法获取网页代码，由于bs4可以自动解码URL的编码，所以此处不需要decode
    html = bs(page, "html.parser")  # 使用html解析器
    wy_title = html.find(class_="post_title")  # 获取新闻网页中的title信息
    detail["title"] = wy_title.text  # 将新闻标题以文本形式存入detail字典中的相应键值中
    news_content = html.find(class_="post_body")  # 使用find方法，获取新闻网页中的article信息
    detail["artibody"] = news_content.text
    detail['news_type'] = 15
    detail['date'] = wy_newsDate
    detail['spider'] = config.wy_spider
    detail["from"] = news['source']
    return detail


def save_content(detail):
    news_dir = config.wy_dir + config.now_date + '\\'
    dir_is_Exists = os.path.exists(news_dir)
    if not dir_is_Exists:
        os.makedirs(news_dir) 
    fp = open(news_dir + detail['id']  +'.txt', 'w+', encoding = 'UTF-8')
    fp.write(json.dumps(detail['artibody'], ensure_ascii = False))
    fp.close()
    
def news_process(wy_news, wy_news_url, wy_newsDate):
    try:
        wy_content = news_content_process(wy_news, wy_news_url, wy_newsDate)
        # 存数据库
        db.insert_news_info(wy_content)
        #存储模块 保存到txt
        save_content(wy_content)
    except Exception as e:
        print(wy_news_url)
        print(e)
        
def check_same_date(newsDate):
    if config.now_date == newsDate:
        return True
    else:
        return False

def main():
    #使用ScalableBloomFilter模块，对获取的URL进行去重处理
    urlbloomfilter = ScalableBloomFilter(initial_capacity=100, error_rate=0.001, mode=ScalableBloomFilter.LARGE_SET_GROWTH)
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
                wy_content = data.content.decode(encoding='UTF-8')
                data_jsons = json.loads(wy_content.replace('data_callback(', '').replace(')', '').encode())
                for wy_news in data_jsons:
                    newsDateArray = time.strptime(wy_news['time'], "%m/%d/%Y %H:%M:%S")
                    wy_newsDate = time.strftime('%Y-%m-%d', newsDateArray)
                    if not check_same_date(wy_newsDate):
                       go_on = False
                       break
                
                    wy_news_url = wy_news["docurl"] 
                    if wy_news_url not in urlbloomfilter:
                        urlbloomfilter.add(wy_news_url) #将爬取过的URL放入urlbloomfilter中
                        news_process(wy_news, wy_news_url, wy_newsDate)
            page += 1
        
if __name__ == '__main__':
    main()