# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 21:43:34 2021

@author: styra
"""
import datetime
import os


news_dir = 'E:\\newslist' + os.sep 

now_date = datetime.date.today().strftime('%Y-%m-%d')

logging_path = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'logging.conf'
project_path = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], "..")) + os.sep


# tencentfact
tencentfact_sign = 'sgn51n6r6q97o6g3'
tencentfact_key = 'jzhotdata'
tencentfact_url = 'https://vp.fact.qq.com/loadmore?artnum=0&token={}&page={}&stopic=&_={}&callback=jsonp'
tencentfact_news_url = 'https://vp.fact.qq.com/article?id={}'
tencentfact_dir = news_dir + 'TencentFact' + os.sep 
tencentfact_spider = 'tencentFactSpider'


# wy
wy_dir = news_dir + 'wy' + os.sep 
wy_url_list = {'domestic':['https://news.163.com/special/cm_guonei{}/?callback=data_callback'],
    'internation':['https://news.163.com/special/cm_guoji{}/?callback=data_callback'],
    'tech':['https://tech.163.com/special/00097UHL/tech_datalist{}.js?callback=data_callback'],
    'society':['http://temp.163.com/special/00804KVA/cm_shehui{}.js?callback=data_callback']
}
wy_spider = 'wySpider'


# sina
sina_dir = news_dir + 'sina' + os.sep 
sina_url = 'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&k=&num=50&page='
sina_spider = 'sinaSpider'


# weibo
weibo_config = os.path.split(os.path.realpath(__file__))[0] + os.sep + 'weibo-config.json'
weibo_spider = news_dir + 'weibo' + os.sep 
weibo_user_path = weibo_spider + 'users.csv'
weibo_profile_url = 'https://m.weibo.cn/profile/'
weibo_article_url = 'https://m.weibo.cn/status/'

#piyao_org
piyao_org_dir = news_dir + 'piyao_org' + os.sep 
piyao_org_url = 'https://dawa.news.cn/nodeart/page?nid=11241459&pgnum={}&cnt=16&attr=&tp=1&orderby=1&callback='
piyao_org_spider = 'piyaoOrgSpider'

#sltg
version = '20220412224218'
w2dic_path = project_path + 'result' + os.sep + 'w2dic.npy'
lstm_path = project_path + 'result' + os.sep + 'lstm_attention' + os.sep + version + os.sep + 'discriminator.h5'
stop_words_path = project_path + 'data' + os.sep + 'stop_words.txt'

