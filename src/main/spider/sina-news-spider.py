# -*- coding: utf-8 -*-
"""
Created on Mon Nov 8 16:47:09 2021

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

def news_content_process(news, newsDate):
    detail = {}  # 创建一个字典，存放URL、title、newstime等信息
    detail["id"] = news['docid'][-7:]
    detail["url"] = news['url']  # 将URL时间存入detail字典中的相应键值中
    page = requests.get(news['url']).content  # 使用requests.get方法获取网页代码，由于bs4可以自动解码URL的编码，所以此处不需要decode
    html = bs(page, "html.parser")  # 使用html解析器
    title = html.find(class_="main-title")  # 获取新闻网页中的title信息，此处网页中只有一个“class=main-title”，所以使用find即可
    detail["title"] = title.text  # 将新闻标题以文本形式存入detail字典中的相应键值中
    news_content = html.find(class_="article")  # 使用find方法，获取新闻网页中的article信息
    detail["artibody"] = news_content.text
    date_source = html.find(class_="date-source")  # 使用find方法，获取新闻网页中的date-source信息
    # 由于不同的新闻详情页之间使用了不同的标签元素，直接抽取可能会报错，所以此处使用判断语句来进行区分爬取
    if date_source.a:  # 判断date-source节点中是否包含有'a'元素
        detail["from"] = date_source.a.text  # 抽取'a'标签中新闻来源信息
    else:
        detail["from"] = date_source("span")[1].text  # 抽取'span'标签中包含的新闻来源信息
    detail['news_type'] = 15
    detail['date'] = newsDate
    detail['spider'] = config.sina_spider
    return detail

def save_content(detail):
    news_dir = config.sina_dir + config.now_date + '\\'
    dir_is_Exists = os.path.exists(news_dir)
    if not dir_is_Exists:
        os.makedirs(news_dir) 
    fp = open(news_dir + detail['id'] +'.txt', 'w+', encoding='UTF-8')
    fp.write(json.dumps(detail['artibody'], ensure_ascii=False))
    fp.close()
    
def news_process(news, newsDate):
    try:
        sina_content = news_content_process(news, newsDate)
        # 存数据库
        db.insert_news_info(sina_content)
        #存储模块 保存到txt
        save_content(sina_content)
    except Exception as e:
        print(news["url"])
        print(e)
    
def check_date(newsDate):
    if config.now_date == newsDate:
        return True
    else:
        return False

def main():
    #使用ScalableBloomFilter模块，对获取的URL进行去重处理
    urlbloomfilter = ScalableBloomFilter(initial_capacity=100, error_rate=0.001, mode=ScalableBloomFilter.LARGE_SET_GROWTH)
    page = 1 #设置爬虫初始爬取的页码
    #使用BeautifulSoup抽取模块和存储模块
    #设置爬取页面的上限，
    go_on = True
    while go_on :
        #以API为index开始获取url列表
        data = requests.get(config.sina_url + str(page)) #拼接URL，并获取索引页面信息
        if data.status_code == 200:  #当请求页面返回200（代表正确）时，获取网页数据
            #将获取的数据json化
            data_json = json.loads(data.content)
            newsList = data_json.get("result").get("data") #获取result节点下data节点中的数据，此数据为新闻详情页的信息
            #从新闻详情页信息列表news中，使用for循环遍历每一个新闻详情页的信息
            for news in newsList:
                newsDateInt = int(news['ctime'])
                newsDateArray = time.localtime(newsDateInt)
                newsDate = time.strftime('%Y-%m-%d', newsDateArray)
                if not check_date(newsDate) :
                    go_on = False
                    break
                
                # 查重，从new中提取URL，并利用ScalableBloomFilter查重
                news_url = news["url"]
                if news_url not in urlbloomfilter:
                    urlbloomfilter.add(news_url) #将爬取过的URL放入urlbloomfilter中
                    news_process(news, newsDate)
            page+=1 #页码自加1
        
if __name__ == '__main__':
    main()