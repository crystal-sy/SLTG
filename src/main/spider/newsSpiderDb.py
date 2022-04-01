# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:36:34 2021

@author: styra
"""
import pymysql

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
    
    percent = detail['detection_percent']
    detectionType = getDetectionType(percent)
    sql = """INSERT INTO sys_news (news_id, news_title, news_url, detection_percent, detection_type,
            news_theme, news_date, news_spider, news_from, create_time) 
            value ( '{id}', '{title}', '{url}', '{detection_percent}', '{detection_type}', 
            '{news_theme}', '{date}', '{spider}', '{news_from}', 
            now())""".format(id = detail['id'], 
                title = detail['title'], url = detail['url'], 
                detection_percent = percent, detection_type = detectionType,
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
    
    percent = detail['detection_percent']
    detectionType = getDetectionType(percent)
    sql = """INSERT INTO sys_news (news_id, news_title, have_title, news_url, news_text,
            news_theme, detection_percent, detection_type, news_date, news_spider, news_from, is_file, create_time) 
            value ( '{id}', '{title}', '{haveTitle}', '{url}', '{text}', '{news_theme}', '{detection_percent}',
            '{detection_type}', '{date}', '{spider}', '{news_from}', '1', now())""".format(id = detail['id'], 
                title = topics, haveTitle = titleFlag, url = detail['url'], 
                text = detail['text'][:5000], news_theme = detail['news_theme'], 
                detection_percent = percent, detection_type = detectionType, date = detail['created_at'], 
                spider = detail['spider'], news_from = detail['source'])
    return execute_update_sql(sql)

def insert_news_comment(detail):
    if detail == {}:
        return 0
    
    sql = """select comment_id from sys_news_comment 
            where comment_id = '{id}'""".format(id = detail['id'])
    result = execute_query_sql(sql)
    if result != '' :
        return 0
    
    sql = """INSERT INTO sys_news_comment (comment_id, news_id, user_id, user_name,
            comment_text, root_id, comment_time, like_count, create_time) 
            value ( '{id}', '{news_id}', '{user_id}', '{user_name}', 
                   '{text}', '{root_id}', '{comtent_time}', '{count}',
            now())""".format(id = detail['id'], news_id = detail['news_id'], 
                user_id = detail['user_id'], user_name = detail['user_name'], 
                text = detail['text'][:5000], root_id = detail['root_id'], 
                comtent_time = detail['created_at'], count = detail['like_count'])
    return execute_update_sql(sql)

def getDetectionType(detectionPercent) :
    if detectionPercent == '':
        return "5"
    
    percent = float(detectionPercent.replace('%', ''))
    if (percent > 90) :
        return "0" # 虚假
    elif (percent > 70) :
        return "1" # 疑似诈骗
    elif (percent > 50) :
        return "4" # 分情况
    elif (percent > 30) :
        return "3" # 待定
    else :
        return "2" # 真实
    

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
    and detection_percent {symbol} "90%"  """.format(symbol=symbol)
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