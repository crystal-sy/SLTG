# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 22:51:24 2022

@author: styra
"""
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from time import sleep
import random
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)
from spider import newsSpiderDb as db

import requests
from config import sltg_config as sltg_config
import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(sltg_config.logging_path)
logger = logging.getLogger('spider')

root_path = project_path + '/data/dataset/weibo/'
file_name = 'weibo'

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'
cookie = '_T_WM=30936573057; WEIBOCN_FROM=1110006030; XSRF-TOKEN=3c6c80; loginScene=103012; SCF=Amu9q24QfwkSKJ2YpbUgi7Ijv89LPR3lJfvA96AecuYm_JKDALSDr80TFkeQR30jMmmSoc-73-Y87kMlQRt6FVQ.; SUB=_2A25PTczXDeRhGeFJ71sY-CfLzz2IHXVssdSfrDV6PUJbktB-LXfakW1Nf6LYyxjoRBE17EqV2CoGrBx0vUfvg5a9; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WF8NrJP9U8ml_261OPRzA1b5JpX5KzhUgL.FoMNSh.41h.NSh22dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMNS0B41Kn4S0Bp; SSOLoginState=1648999560; ALF=1651591560; MLOGIN=1; mweibo_short_token=3e39921508; M_WEIBOCN_PARAMS=oid%3D4754282278294412%26luicode%3D20000061%26lfid%3D4754282278294412%26uicode%3D20000061%26fid%3D4754282278294412'
headers = {'User_Agent': user_agent, 'Cookie': cookie}

def get_weibo_comments_cookie(id, cur_count, max_count, max_id, comment_list):
    """
    :weibo standardlized weibo
    :cur_count  已经下载的评论数
    :max_count 最大允许下载数
    :max_id 微博返回的max_id参数
    :on_downloaded 下载完成时的实例方法回调
    """
    if cur_count >= max_count:
        return

    url = "https://m.weibo.cn/comments/hotflow?id={}&mid={}&max_id_type=0".format(id, id)
    if max_id:
        url = "https://m.weibo.cn/comments/hotflow?id={}&mid={}&max_id={}&max_id_type=0".format(id, id, max_id)
    
    req = requests.get(
        url,
        headers=headers,
    )
    json = None
    error = False
    try:
        json = req.json()
    except Exception :
        logger.info(u'cookie未能抓取评论 微博id:{id}'.format(id=id))
        #没有cookie会抓取失败
        #微博日期小于某个日期的用这个url会被403 需要用老办法尝试一下
        error = True

    if error:
        #最大好像只能有50条 TODO: improvement
        get_weibo_comments_nocookie(id, 0, max_count, 0, comment_list)
        return

    data = json.get('data')
    if not data:
        #新接口没有抓取到的老接口也试一下
        get_weibo_comments_nocookie(id, 0, max_count, 0, comment_list)
        return

    comments = data.get('data')
    count = len(comments)
    if count == 0:
        #没有了可以直接跳出递归
        return

    comment_list.append(comments)

    #随机睡眠一下
    if max_count % 40 == 0:
        sleep(random.randint(1, 5))

    cur_count += count
    max_id = data.get('max_id')

    if max_id == 0:
        return

    get_weibo_comments_cookie(id, cur_count, max_count, max_id, comment_list)
    
def get_weibo_comments_nocookie(id, cur_count, max_count, page, comment_list):
    """
    :weibo standardlized weibo
    :cur_count  已经下载的评论数
    :max_count 最大允许下载数
    :max_id 微博返回的max_id参数
    """
    if cur_count >= max_count:
        return
    url = "https://m.weibo.cn/api/comments/show?id={id}&page={page}".format(
        id=id, page=page)
    req = requests.get(url)
    json = None
    try:
        json = req.json()
    except Exception :
        #没有cookie会抓取失败
        logger.info(u'nocookie未能抓取评论 微博id:{id}'.format(id=id))
        return

    data = json.get('data')
    if not data:
        return
    comments = data.get('data')
    count = len(comments)
    if count == 0:
        #没有了可以直接跳出递归
        return

    comment_list.append(comments)

    cur_count += count
    page += 1

    #随机睡眠一下
    if page % 2 == 0:
        sleep(random.randint(1, 5))

    req_page = data.get('max')

    if req_page == 0:
        return

    if page >= req_page:
        return
    get_weibo_comments_nocookie(id, cur_count, max_count, page, comment_list)

def parse_sqlite_comment(comment, weibo_id):
    if not comment:
        return
    
    sqlite_comment = OrderedDict()
    sqlite_comment["id"] = comment['id']
    sqlite_comment["news_id"] = weibo_id
    sqlite_comment["user_id"] = comment['user']['id']
    sqlite_comment["user_name"] = comment['user']['screen_name']
    sqlite_comment["text"] = try_get_text(comment['text'])
    if sqlite_comment["text"] == '':
        return ''
    
    try_get_value('root_id', 'rootid', sqlite_comment, comment)
    try_get_value('created_at', 'created_at', sqlite_comment,
                        comment)
    sqlite_comment["created_at"] = standardize_date(
        sqlite_comment["created_at"], '%Y-%m-%d %H:%M:%S')
    
    if sqlite_comment['id'] == sqlite_comment['root_id']:
        sqlite_comment["like_count"] = try_get_like_count(comment)
    else:
        sqlite_comment["like_count"] = 0
    return sqlite_comment

def standardize_date(created_at, fromat = '%Y-%m-%d'):
    """判断日期格式是否正确"""
    try:
        datetime.strptime(created_at, '%Y-%m-%d')
        return created_at
    except ValueError:
        """标准化微博发布时间"""
        if u'刚刚' in created_at:
            created_at = datetime.now().strftime(fromat)
        elif u'分钟' in created_at:
            minute = created_at[:created_at.find(u'分钟')]
            minute = timedelta(minutes=int(minute))
            created_at = (datetime.now() - minute).strftime(fromat)
        elif u'小时' in created_at:
            hour = created_at[:created_at.find(u'小时')]
            hour = timedelta(hours=int(hour))
            created_at = (datetime.now() - hour).strftime(fromat)
        elif u'昨天' in created_at:
            day = timedelta(days=1)
            created_at = (datetime.now() - day).strftime(fromat)
        else:
            created_at = created_at.replace('+0800 ', '')
            temp = datetime.strptime(created_at, '%c')
            created_at = datetime.strftime(temp, fromat)
        return created_at   
        
def try_get_text(text):
    if text == '':
        return ''
    
    result = text.replace('/n', '')
    try:
        while result.index('<span') > -1:
            start = result.index('<span')
            end = result.index('</span>') + 7
            result = result.replace(result[start:end], '')
    except Exception :
        logger.info(u'表情处理完成:%s', result)
        
    try:
        while result.index('<a ') > -1:
            start = result.index('<a ')
            end = result.index('>') + 1
            result = result.replace(result[start:end], '').replace('</a>', '')
    except Exception :
        logger.info(u'@某人处理完成:%s', result)
    return result.replace("'","")
        
def try_get_like_count(json):
    count = 0
    value1 = json.get('like_count')
    value2 = json.get('like_counts')
    if value1 is not None:
        count = value1
    elif value2 is not None:
        count = value2
    return count

def try_get_value(source_name, target_name, dict, json):
    dict[source_name] = ''
    value = json.get(target_name)
    if value:
        dict[source_name] = value
    
def get_comment(id) :
    comment_list = []
    get_weibo_comments_cookie(id, 0, 1000, None, comment_list)
    if not comment_list or len(comment_list) == 0:
        return
    
    for comments in comment_list:
        for comment in comments:
            insert_comment(comment, id)
            if comment.get('comments') is not None:
                childs = comment['comments'] 
                if childs is not None and childs != False:
                    for child in childs :
                        insert_comment(child, id)
                    
def insert_comment(comment, id):
    data = parse_sqlite_comment(comment, id)
    if data != '':
        try:
            db.insert_news_comment(data)
        except Exception :
            logger.error(u'评论插入异常:%s', data['text'])          

with open(root_path + file_name +".train", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        try:
            get_comment(tid)
        except Exception :
            logger.error(u'train 评论爬取异常:%s', tid) 
        

with open(root_path + file_name +".dev", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        try:
            get_comment(tid)
        except Exception :
            logger.error(u'dev 评论爬取异常:%s', tid) 


with open(root_path + file_name +".test", 'r', encoding='utf-8') as input:
    for line in input.readlines():
        tid, content, label = line.strip().split("\t")
        try:
            get_comment(tid)
        except Exception :
            logger.error(u'test 评论爬取异常:%s', tid)