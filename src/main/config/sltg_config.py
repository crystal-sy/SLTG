# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:43:34 2021

@author: styra
"""
import datetime
import os


project_path = 'D:\sycode\SLTG\src\main'
news_dir = 'D:\\hscode\\newslist\\'

now_date = datetime.date.today().strftime('%Y-%m-%d')

logging_path = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'logging.conf'


# tencentfact
tencentfact_sign = 'sgn51n6r6q97o6g3'
tencentfact_key = 'jzhotdata'
tencentfact_url = 'https://vp.fact.qq.com/loadmore?artnum=0&token={}&page={}&stopic=&_={}&callback=jsonp'
tencentfact_news_url = 'https://vp.fact.qq.com/article?id={}'
tencentfact_dir = news_dir + 'TencentFact\\'
tencentfact_spider = 'tencentFactSpider'


# wy
wy_dir = news_dir + 'wy\\'
wy_url_list = {'domestic':['https://news.163.com/special/cm_guonei{}/?callback=data_callback'],
    'internation':['https://news.163.com/special/cm_guoji{}/?callback=data_callback'],
    'tech':['https://tech.163.com/special/00097UHL/tech_datalist{}.js?callback=data_callback'],
    'society':['http://temp.163.com/special/00804KVA/cm_shehui{}.js?callback=data_callback']
}
wy_spider = 'wySpider'


# sina
sina_dir = news_dir + 'sina\\'
sina_url = 'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=50&page='
sina_spider = 'sinaSpider'


# weibo
weibo_config = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'weibo-config.json1'