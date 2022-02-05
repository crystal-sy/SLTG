# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 22:36:34 2021

@author: styra
"""
import pymysql

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
    
    sql = """INSERT INTO sys_news (news_id, news_title, news_url, 
            news_theme, news_date, news_spider, news_from, create_time) 
            value ( '{id}', '{title}', '{url}', '{news_theme}', '{date}', 
            '{spider}', '{news_from}', now())""".format(id = detail['id'], 
                title = detail['title'], url = detail['url'], 
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
            news_theme, news_date, news_spider, news_from, is_file, create_time) 
            value ( '{id}', '{title}', '{haveTitle}', '{url}', '{text}', '{news_theme}', '{date}', 
            '{spider}', '{news_from}', '1', now())""".format(id = detail['id'], 
                title = topics, haveTitle = titleFlag, url = detail['url'], 
                text = detail['text'][:5000], news_theme = detail['news_theme'], date = detail['created_at'], 
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
