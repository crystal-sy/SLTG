# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:36:34 2021

@author: styra
"""
import pymysql

import sys
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
sys.path.append(project_path)

from config import sltg_config as config
import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(config.logging_path)
logger = logging.getLogger('spider')

mysql_config = {
    'db': 'sltg-vue',
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'charset': 'utf8mb4'
}

def query_news_knowledge_last_date(platform):
    sql = """SELECT news_date FROM sys_news_knowledge 
            WHERE news_spider = '{platform}'
            ORDER BY news_date DESC LIMIT 1""".format(platform = platform)
    return execute_query_sql(sql)

def insert_news_knowledge(detail):
    if detail == {}:
        return 0
    
    sql = """select news_id from sys_news_knowledge 
            where news_id = '{id}'""".format(id = detail['id'])
    result = execute_query_sql(sql)
    if result != '' :
        return 0
    
    sql = """INSERT INTO sys_news_knowledge (news_id, news_title, news_url, 
            news_theme, detection_type, detection_result, news_date, 
            news_spider, news_from, original_url, create_time) value ( 
            '{id}', '{title}', '{url}', '{news_theme}', '{detection_type}', 
            '{result}', '{date}', '{spider}', '{news_from}',
            '{original_url}', now())""".format(id = detail['id'], title = detail['title'],
                url = detail['url'], news_theme = detail['news_theme'], 
                detection_type = detail['detection_type'], result = detail['result'], 
                date = detail['date'], spider = detail['spider'],
                news_from = detail['from'], original_url = detail['original_url'])
    return execute_update_sql(sql)

def insert_news_info(detail):
    if detail == {}:
        return 0
    
    sql = """select news_id from sys_news 
            where news_id = '{id}'""".format(id = detail['id'])
    result = execute_query_sql(sql)
    if result != '' :
        return 0
    
    sql = """INSERT INTO sys_news (news_id, news_title, news_url, detection_percent,
            news_theme, news_date, news_spider, news_from, create_time) 
            value ( '{id}', '{title}', '{url}', '{detection_percent}', 
            '{news_theme}', '{date}', '{spider}', '{news_from}', 
            now())""".format(id = detail['id'], 
                title = detail['title'], url = detail['url'], 
                detection_percent = detail['detection_percent'], 
                news_theme = detail['news_theme'], date = detail['date'], 
                spider = detail['spider'], news_from = detail['from'])
    return execute_update_sql(sql)

def insert_news_info_weibo(detail):
    if detail == {}:
        return 0
    
    sql = """select news_id from sys_news 
            where news_id = '{id}'""".format(id = detail['id'])
    result = execute_query_sql(sql)
    if result != '' :
        return 0
    
    topics = detail['topics']
    titleFlag = 0
    if topics == '' :
        topics = detail['text'][:50] + "..."
        titleFlag = 1
    
    sql = """INSERT INTO sys_news (news_id, news_title, have_title, news_url, news_text,
            news_theme, detection_percent, news_date, news_spider, news_from, is_file, create_time) 
            value ( '{id}', '{title}', '{haveTitle}', '{url}', '{text}', '{news_theme}', '{detection_percent}',
            '{date}', '{spider}', '{news_from}', '1', now())""".format(id = detail['id'], 
                title = topics, haveTitle = titleFlag, url = detail['url'], 
                text = detail['text'][:5000], news_theme = detail['news_theme'], 
                detection_percent = detail['detection_percent'], date = detail['created_at'], 
                spider = detail['spider'], news_from = detail['source'])
    return execute_update_sql(sql)

def execute_query_sql(sql):
    connection = pymysql.connect(**mysql_config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            for row in cursor.fetchall() :
                return row[0]
    finally:
        connection.close()
    return ''

def execute_update_sql(sql):
    connection = pymysql.connect(**mysql_config)
    try:
        with connection.cursor() as cursor:
            cnt = cursor.execute(sql)
            connection.commit()
            return cnt
    finally:
        connection.close()
    return 0

def query_news_keyword_per_month_real(type_news):
    symbol = ""
    resultDict = ''
    connection = pymysql.connect(**mysql_config)
    if type_news is True:
        symbol = "<="
    else:
        symbol = ">"
    sql ="""select news_theme as keywords from sys_news WHERE DATE_SUB(CURDATE(), INTERVAL 30 DAY) <= news_date
    and detection_percent {symbol} "50%"  """.format(symbol=symbol)
    try:
        with connection.cursor() as cursor:
            # 执行SQL语句
            cursor.execute(sql)
            for row in cursor.fetchall() :
                row = ' '.join(row)
                resultDict = resultDict + row + ','
    except:
        logger.error("Error: unable to fetch data")
    finally:
        connection.close() 
    return resultDict