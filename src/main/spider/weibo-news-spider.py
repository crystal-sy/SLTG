#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import copy
import csv
import json
import logging
import logging.config
import math
import os
import random
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, timedelta
from time import sleep

import requests
from lxml import etree
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging_path = os.path.split(
    os.path.realpath(__file__))[0] + os.sep + 'logging.conf'
logging.config.fileConfig(logging_path)
logger = logging.getLogger('weibo')


class Weibo(object):
    def __init__(self, config):
        """Weibo类初始化"""
        self.validate_config(config)
        self.filter = config[
            'filter']  # 取值范围为0、1,程序默认值为0,代表要爬取用户的全部微博,1代表只爬取用户的原创微博
        self.remove_html_tag = config[
            'remove_html_tag']  # 取值范围为0、1, 0代表不移除微博中的html tag, 1代表移除
        since_date = config['since_date']
        if isinstance(since_date, int):
            since_date = date.today() - timedelta(since_date)
        since_date = str(since_date)
        self.since_date = since_date  # 起始时间，即爬取发布日期从该值到现在的微博，形式为yyyy-mm-dd
        self.start_page = config.get('start_page',
                                     1)  # 开始爬的页，如果中途被限制而结束可以用此定义开始页码
        self.write_mode = config[
            'write_mode']  # 结果信息保存类型，为list形式，可包含csv、mysql两种类型
        self.download_comment = config['download_comment']  #1代表下载评论,0代表不下载
        self.comment_max_download_count = config[
            'comment_max_download_count']  #如果设置了下评论，每条微博评论数会限制在这个值内
        self.result_dir_name = config.get(
            'result_dir_name', 0)  # 结果目录名，取值为0或1，决定结果文件存储在用户昵称文件夹里还是用户id文件夹里
        cookie = config.get('cookie')  # 微博cookie，可填可不填
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
        self.headers = {'User_Agent': user_agent, 'Cookie': cookie}

        self.mysql_config = config.get('mysql_config')  # MySQL数据库连接配置，可以不填

        user_id_list = config['user_id_list']
        query_list = config.get('query_list') or []
        if isinstance(query_list, str):
            query_list = query_list.split(',')
        self.query_list = query_list
        if not isinstance(user_id_list, list):
            if not os.path.isabs(user_id_list):
                user_id_list = os.path.split(
                    os.path.realpath(__file__))[0] + os.sep + user_id_list
            self.user_config_file_path = user_id_list  # 用户配置文件路径
            user_config_list = self.get_user_config_list(user_id_list)
        else:
            self.user_config_file_path = ''
            user_config_list = [{
                'user_id': user_id,
                'since_date': self.since_date,
                'query_list': query_list
            } for user_id in user_id_list]
        self.user_config_list = user_config_list  # 要爬取的微博用户的user_config列表
        self.user_config = {}  # 用户配置,包含用户id和since_date
        self.start_date = ''  # 获取用户第一条微博时的日期
        self.query = ''
        self.user = {}  # 存储目标微博用户信息
        self.got_count = 0  # 存储爬取到的微博数
        self.weibo = []  # 存储爬取到的所有微博信息
        self.weibo_id_list = []  # 存储爬取到的所有微博id

    def validate_config(self, config):
        """验证配置是否正确"""

        # 验证filter、
        argument_list = [
            'filter'
        ]
        for argument in argument_list:
            if config[argument] != 0 and config[argument] != 1:
                logger.warning(u'%s值应为0或1,请重新输入', config[argument])
                sys.exit()

        # 验证since_date
        since_date = config['since_date']
        if (not self.is_date(str(since_date))) and (not isinstance(
                since_date, int)):
            logger.warning(u'since_date值应为yyyy-mm-dd形式或整数,请重新输入')
            sys.exit()

        # 验证query_list
        query_list = config.get('query_list') or []
        if (not isinstance(query_list, list)) and (not isinstance(
                query_list, str)):
            logger.warning(u'query_list值应为list类型或字符串,请重新输入')
            sys.exit()

        # 验证write_mode
        write_mode = ['csv', 'json', 'mysql']
        if not isinstance(config['write_mode'], list):
            sys.exit(u'write_mode值应为list类型')
        for mode in config['write_mode']:
            if mode not in write_mode:
                logger.warning(
                    u'%s为无效模式，请从csv、json、mysql中挑选一个或多个作为write_mode',
                    mode)
                sys.exit()

        # 验证user_id_list
        user_id_list = config['user_id_list']
        if (not isinstance(user_id_list,
                           list)) and (not user_id_list.endswith('.txt')):
            logger.warning(u'user_id_list值应为list类型或txt文件路径')
            sys.exit()
        if not isinstance(user_id_list, list):
            if not os.path.isabs(user_id_list):
                user_id_list = os.path.split(
                    os.path.realpath(__file__))[0] + os.sep + user_id_list
            if not os.path.isfile(user_id_list):
                logger.warning(u'不存在%s文件', user_id_list)
                sys.exit()

        comment_max_count = config['comment_max_download_count']
        if (not isinstance(comment_max_count, int)):
            logger.warning(u'最大下载评论数应为整数类型')
            sys.exit()
        elif (comment_max_count < 0):
            logger.warning(u'最大下载数应该为正整数')
            sys.exit()

    def is_date(self, since_date):
        """判断日期格式是否正确"""
        try:
            datetime.strptime(since_date, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def get_json(self, params):
        """获取网页中json数据"""
        url = 'https://m.weibo.cn/api/container/getIndex?'
        r = requests.get(url,
                         params=params,
                         headers=self.headers,
                         verify=False)
        return r.json()

    def get_weibo_json(self, page):
        """获取网页中微博json数据"""
        params = {
            'container_ext': 'profile_uid:' + str(self.user_config['user_id']),
            'containerid': '100103type=401&q=' + self.query,
            'page_type': 'searchall'
        } if self.query else {
            'containerid': '107603' + str(self.user_config['user_id'])
        }
        params['page'] = page
        js = self.get_json(params)
        return js

    def user_to_csv(self):
        """将爬取到的用户信息写入csv文件"""
        file_dir = os.path.split(
            os.path.realpath(__file__))[0] + os.sep + 'weibo'
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        file_path = file_dir + os.sep + 'users.csv'
        result_headers = [
            '用户id', '昵称', 
            '微博数', '粉丝数', '关注数', 
            '是否认证', '认证类型', '认证信息'
        ]
        result_data = [[
            v.encode('utf-8') if 'unicode' in str(type(v)) else v
            for v in self.user.values()
        ]]
        self.csv_helper(result_headers, result_data, file_path)



    def user_to_mysql(self):
        """将爬取的用户信息写入MySQL数据库"""
        mysql_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'password',
            'charset': 'utf8mb4'
            }
        # 创建'weibo'数据库
        create_database = """CREATE DATABASE IF NOT EXISTS newsData DEFAULT
        CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"""
        self.mysql_create_database(mysql_config, create_database)
        # 创建'user'表
        create_table = """
        CREATE TABLE IF NOT EXISTS weibo_user_info (
            id varchar(20) NOT NULL,
            screen_name varchar(30),
            statuses_count INT,
            followers_count INT,
            follow_count INT,
            description varchar(400),
            profile_url varchar(200),
            profile_image_url varchar(200),
            avatar_hd varchar(200),
            urank INT,
            mbrank INT,
            verified BOOLEAN DEFAULT 0,
            verified_type INT,
            verified_reason varchar(140),
            PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
        self.mysql_create_table(mysql_config, create_table)
        self.mysql_insert(mysql_config, 'weibo_user_info', [self.user])
        logger.info(u'%s信息写入MySQL数据库完毕', self.user['screen_name'])

    def user_to_database(self):
         """将用户信息写入文件/数据库"""
         self.user_to_csv()
         if 'mysql' in self.write_mode:
             self.user_to_mysql()

    def get_user_info(self):
        """获取用户信息"""
        params = {'containerid': '100505' + str(self.user_config['user_id'])}
        js = self.get_json(params)
        if js['ok']:
            info = js['data']['userInfo']
            user_info = OrderedDict()
            user_info['id'] = self.user_config['user_id']
            user_info['screen_name'] = info.get('screen_name', '')
            params = {
                'containerid':
                '230283' + str(self.user_config['user_id']) + '_-_INFO'
            }
            user_info['statuses_count'] = self.string_to_int(
                info.get('statuses_count', 0))
            user_info['followers_count'] = self.string_to_int(
                info.get('followers_count', 0))
            user_info['follow_count'] = self.string_to_int(
                info.get('follow_count', 0))
            user_info['description'] = info.get('description', '')
            user_info['profile_url'] = info.get('profile_url', '')
            user_info['urank'] = info.get('urank', 0)
            user_info['mbrank'] = info.get('mbrank', 0)
            user_info['verified'] = info.get('verified', False)
            user_info['verified_type'] = info.get('verified_type', -1)
            user_info['verified_reason'] = info.get('verified_reason', '')
            user = self.standardize_info(user_info)
            self.user = user
            self.user_to_database()
            return user
        else:
            logger.info(u"被ban了")
            sys.exit()

    def get_long_weibo(self, id):
        """获取长微博"""
        for i in range(5):
            url = 'https://m.weibo.cn/detail/%s' % id
            html = requests.get(url, headers=self.headers, verify=False).text
            html = html[html.find('"status":'):]
            html = html[:html.rfind('"hotScheme"')]
            html = html[:html.rfind(',')]
            html = '{' + html + '}'
            js = json.loads(html, strict=False)
            weibo_info = js.get('status')
            if weibo_info:
                weibo = self.parse_weibo(weibo_info)
                return weibo
            sleep(random.randint(6, 10))

    def get_location(self, selector):
        """获取微博发布位置"""
        location_icon = 'timeline_card_small_location_default.png'
        span_list = selector.xpath('//span')
        location = ''
        for i, span in enumerate(span_list):
            if span.xpath('img/@src'):
                if location_icon in span.xpath('img/@src')[0]:
                    location = span_list[i + 1].xpath('string(.)')
                    break
        return location

    def get_article_url(self, selector):
        """获取微博中头条文章的url"""
        article_url = ''
        text = selector.xpath('string(.)')
        if text.startswith(u'发布了头条文章'):
            url = selector.xpath('//a/@data-url')
            if url and url[0].startswith('http://t.cn'):
                article_url = url[0]
        return article_url

    def get_topics(self, selector):
        """获取参与的微博话题"""
        span_list = selector.xpath("//span[@class='surl-text']")
        topics = ''
        topic_list = []
        for span in span_list:
            text = span.xpath('string(.)')
            if len(text) > 2 and text[0] == '#' and text[-1] == '#':
                topic_list.append(text[1:-1])
        if topic_list:
            topics = ','.join(topic_list)
        return topics

    def get_at_users(self, selector):
        """获取@用户"""
        a_list = selector.xpath('//a')
        at_users = ''
        at_list = []
        for a in a_list:
            if '@' + a.xpath('@href')[0][3:] == a.xpath('string(.)'):
                at_list.append(a.xpath('string(.)')[1:])
        if at_list:
            at_users = ','.join(at_list)
        return at_users

    def string_to_int(self, string):
        """字符串转换为整数"""
        if isinstance(string, int):
            return string
        elif string.endswith(u'万+'):
            string = string[:-2] + '0000'
        elif string.endswith(u'万'):
            string = float(string[:-1]) * 10000
        elif string.endswith(u'亿'):
            string = float(string[:-1]) * 100000000
        return int(string)

    def standardize_date(self, created_at):
        """标准化微博发布时间"""
        if u'刚刚' in created_at:
            created_at = datetime.now().strftime('%Y-%m-%d')
        elif u'分钟' in created_at:
            minute = created_at[:created_at.find(u'分钟')]
            minute = timedelta(minutes=int(minute))
            created_at = (datetime.now() - minute).strftime('%Y-%m-%d')
        elif u'小时' in created_at:
            hour = created_at[:created_at.find(u'小时')]
            hour = timedelta(hours=int(hour))
            created_at = (datetime.now() - hour).strftime('%Y-%m-%d')
        elif u'昨天' in created_at:
            day = timedelta(days=1)
            created_at = (datetime.now() - day).strftime('%Y-%m-%d')
        else:
            created_at = created_at.replace('+0800 ', '')
            temp = datetime.strptime(created_at, '%c')
            created_at = datetime.strftime(temp, '%Y-%m-%d')
        return created_at

    def standardize_info(self, weibo):
        """标准化信息，去除乱码"""
        for k, v in weibo.items():
            if 'bool' not in str(type(v)) and 'int' not in str(
                    type(v)) and 'list' not in str(
                        type(v)) and 'long' not in str(type(v)):
                weibo[k] = v.replace(u'\u200b', '').encode(
                    sys.stdout.encoding, 'ignore').decode(sys.stdout.encoding)
        return weibo

    def parse_weibo(self, weibo_info):
        weibo = OrderedDict()
        if weibo_info['user']:
            weibo['user_id'] = weibo_info['user']['id']
            weibo['screen_name'] = weibo_info['user']['screen_name']
        else:
            weibo['user_id'] = ''
            weibo['screen_name'] = ''
        weibo['id'] = int(weibo_info['id'])
        weibo['bid'] = weibo_info['bid']
        text_body = weibo_info['text']
        selector = etree.HTML(text_body)
        if self.remove_html_tag:
            weibo['text'] = selector.xpath('string(.)')
        else:
            weibo['text'] = text_body
        weibo['article_url'] = self.get_article_url(selector)
        weibo['location'] = self.get_location(selector)
        weibo['created_at'] = weibo_info['created_at']
        weibo['source'] = weibo_info['source']
        weibo['attitudes_count'] = self.string_to_int(
            weibo_info.get('attitudes_count', 0))
        weibo['comments_count'] = self.string_to_int(
            weibo_info.get('comments_count', 0))
        weibo['reposts_count'] = self.string_to_int(
            weibo_info.get('reposts_count', 0))
        weibo['topics'] = self.get_topics(selector)
        weibo['at_users'] = self.get_at_users(selector)
        return self.standardize_info(weibo)

    def print_user_info(self):
        """打印用户信息"""
        logger.info('+' * 100)
        logger.info(u'用户信息')
        logger.info(u'用户id：%s', self.user['id'])
        logger.info(u'用户昵称：%s', self.user['screen_name'])
        logger.info(u'微博数：%d', self.user['statuses_count'])
        logger.info(u'粉丝数：%d', self.user['followers_count'])
        logger.info(u'关注数：%d', self.user['follow_count'])
        logger.info(u'url：https://m.weibo.cn/profile/%s', self.user['id'])
        if self.user.get('verified_reason'):
            logger.info(self.user['verified_reason'])
        logger.info(self.user['description'])
        logger.info('+' * 100)

    def print_one_weibo(self, weibo):
        """打印一条微博"""
        try:
            logger.info(u'微博id：%d', weibo['id'])
            logger.info(u'微博正文：%s', weibo['text'])
            logger.info(u'微博位置：%s', weibo['location'])
            logger.info(u'发布时间：%s', weibo['created_at'])
            logger.info(u'发布工具：%s', weibo['source'])
            logger.info(u'点赞数：%d', weibo['attitudes_count'])
            logger.info(u'评论数：%d', weibo['comments_count'])
            logger.info(u'转发数：%d', weibo['reposts_count'])
            logger.info(u'话题：%s', weibo['topics'])
            logger.info(u'@用户：%s', weibo['at_users'])
            logger.info(u'url：https://m.weibo.cn/detail/%d', weibo['id'])
        except OSError:
            pass

    def print_weibo(self, weibo):
        """打印微博，若为转发微博，会同时打印原创和转发部分"""
        if weibo.get('retweet'):
            logger.info('*' * 100)
            logger.info(u'转发部分：')
            self.print_one_weibo(weibo['retweet'])
            logger.info('*' * 100)
            logger.info(u'原创部分：')
        self.print_one_weibo(weibo)
        logger.info('-' * 120)

    def get_one_weibo(self, info):
        """获取一条微博的全部信息"""
        try:
            weibo_info = info['mblog']
            weibo_id = weibo_info['id']
            retweeted_status = weibo_info.get('retweeted_status')
            is_long = True if weibo_info.get(
                'pic_num') > 9 else weibo_info.get('isLongText')
            if retweeted_status and retweeted_status.get('id'):  # 转发
                retweet_id = retweeted_status.get('id')
                is_long_retweet = retweeted_status.get('isLongText')
                if is_long:
                    weibo = self.get_long_weibo(weibo_id)
                    if not weibo:
                        weibo = self.parse_weibo(weibo_info)
                else:
                    weibo = self.parse_weibo(weibo_info)
                if is_long_retweet:
                    retweet = self.get_long_weibo(retweet_id)
                    if not retweet:
                        retweet = self.parse_weibo(retweeted_status)
                else:
                    retweet = self.parse_weibo(retweeted_status)
                retweet['created_at'] = self.standardize_date(
                    retweeted_status['created_at'])
                weibo['retweet'] = retweet
            else:  # 原创
                if is_long:
                    weibo = self.get_long_weibo(weibo_id)
                    if not weibo:
                        weibo = self.parse_weibo(weibo_info)
                else:
                    weibo = self.parse_weibo(weibo_info)
            weibo['created_at'] = self.standardize_date(
                weibo_info['created_at'])
            return weibo
        except Exception as e:
            logger.exception(e)

    def get_weibo_comments(self, weibo, max_count, on_downloaded):
        """
        :weibo standardlized weibo
        :max_count 最大允许下载数
        :on_downloaded 下载完成时的实例方法回调
        """
        if weibo['comments_count'] == 0:
            return

        logger.info(u'正在下载评论 微博id:{id}正文:{text}'.format(id=weibo['id'],
                                                        text=weibo['text']))
        self._get_weibo_comments_cookie(weibo, 0, max_count, None,
                                        on_downloaded)

    def _get_weibo_comments_cookie(self, weibo, cur_count, max_count, max_id,
                                   on_downloaded):
        """
        :weibo standardlized weibo
        :cur_count  已经下载的评论数
        :max_count 最大允许下载数
        :max_id 微博返回的max_id参数
        :on_downloaded 下载完成时的实例方法回调
        """
        if cur_count >= max_count:
            return

        id = weibo["id"]
        params = {"mid": id}
        if max_id:
            params["max_id"] = max_id

        url = "https://m.weibo.cn/comments/hotflow?max_id_type=0"
        req = requests.get(
            url,
            params=params,
            headers=self.headers,
        )
        json = None
        error = False
        try:
            json = req.json()
        except Exception as e:
            print(e)
            #没有cookie会抓取失败
            #微博日期小于某个日期的用这个url会被403 需要用老办法尝试一下
            error = True

        if error:
            #最大好像只能有50条 TODO: improvement
            self._get_weibo_comments_nocookie(weibo, 0, max_count, 0,
                                              on_downloaded)
            return

        data = json.get('data')
        if not data:
            #新接口没有抓取到的老接口也试一下
            self._get_weibo_comments_nocookie(weibo, 0, max_count, 0,
                                              on_downloaded)
            return

        comments = data.get('data')
        count = len(comments)
        if count == 0:
            #没有了可以直接跳出递归
            return

        if on_downloaded:
            on_downloaded(weibo, comments)

        #随机睡眠一下
        if max_count % 40 == 0:
            sleep(random.randint(1, 5))

        cur_count += count
        max_id = data.get('max_id')

        if max_id == 0:
            return

        self._get_weibo_comments_cookie(weibo, cur_count, max_count, max_id,
                                        on_downloaded)

    def _get_weibo_comments_nocookie(self, weibo, cur_count, max_count, page,
                                     on_downloaded):
        """
        :weibo standardlized weibo
        :cur_count  已经下载的评论数
        :max_count 最大允许下载数
        :max_id 微博返回的max_id参数
        :on_downloaded 下载完成时的实例方法回调
        """
        if cur_count >= max_count:
            return
        id = weibo['id']
        url = "https://m.weibo.cn/api/comments/show?id={id}&page={page}".format(
            id=id, page=page)
        req = requests.get(url)
        json = None
        try:
            json = req.json()
        except Exception as e:
            print(e)
            #没有cookie会抓取失败
            logger.info(u'未能抓取评论 微博id:{id} 内容{text}'.format(
                id=id, text=weibo['text']))
            return

        data = json.get('data')
        if not data:
            return
        comments = data.get('data')
        count = len(comments)
        if count == 0:
            #没有了可以直接跳出递归
            return

        if on_downloaded:
            on_downloaded(weibo, comments)

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
        self._get_weibo_comments_nocookie(weibo, cur_count, max_count, page,
                                          on_downloaded)

    def is_pinned_weibo(self, info):
        """判断微博是否为置顶微博"""
        weibo_info = info['mblog']
        title = weibo_info.get('title')
        if title and title.get('text') == u'置顶':
            return True
        else:
            return False

    def get_one_page(self, page):
        """获取一页的全部微博"""
        try:
            js = self.get_weibo_json(page)
            if js['ok']:
                weibos = js['data']['cards']
                if self.query:
                    weibos = weibos[0]['card_group']
                for w in weibos:
                    if w['card_type'] == 9:
                        wb = self.get_one_weibo(w)
                        if wb:
                            if wb['id'] in self.weibo_id_list:
                                continue
                            created_at = datetime.strptime(
                                wb['created_at'], '%Y-%m-%d')
                            since_date = datetime.strptime(
                                self.user_config['since_date'], '%Y-%m-%d')
                            if created_at < since_date:
                                if self.is_pinned_weibo(w):
                                    continue
                                else:
                                    logger.info(
                                        u'{}已获取{}({})的第{}页{}微博{}'.format(
                                            '-' * 30, self.user['screen_name'],
                                            self.user['id'], page,
                                            '包含"' + self.query +
                                            '"的' if self.query else '',
                                            '-' * 30))
                                    return True
                            if (not self.filter) or (
                                    'retweet' not in wb.keys()):
                                self.weibo.append(wb)
                                self.weibo_id_list.append(wb['id'])
                                self.got_count += 1
                                self.print_weibo(wb)
                            else:
                                logger.info(u'正在过滤转发微博')
            else:
                return True
            logger.info(u'{}已获取{}({})的第{}页微博{}'.format(
                '-' * 30, self.user['screen_name'], self.user['id'], page,
                '-' * 30))
        except Exception as e:
            logger.exception(e)

    def get_page_count(self):
        """获取微博页数"""
        try:
            weibo_count = self.user['statuses_count']
            page_count = int(math.ceil(weibo_count / 10.0))
            return page_count
        except KeyError:
            logger.exception(
                u'程序出错，错误原因可能为以下两者：\n'
                u'1.user_id不正确；\n'
                u'2.此用户微博可能需要设置cookie才能爬取。\n'
                u'解决方案：\n'
                u'请参考\n'
                u'https://github.com/dataabc/weibo-crawler#如何获取user_id\n'
                u'获取正确的user_id；\n'
                u'或者参考\n'
                u'https://github.com/dataabc/weibo-crawler#3程序设置\n'
                u'中的“设置cookie”部分设置cookie信息')

    def get_write_info(self, wrote_count):
        """获取要写入的微博信息"""
        write_info = []
        for w in self.weibo[wrote_count:]:
            wb = OrderedDict()
            for k, v in w.items():
                if k not in ['user_id', 'screen_name', 'retweet']:
                    if 'unicode' in str(type(v)):
                        v = v.encode('utf-8')
                    wb[k] = v
            if not self.filter:
                if w.get('retweet'):
                    wb['is_original'] = False
                    for k2, v2 in w['retweet'].items():
                        if 'unicode' in str(type(v2)):
                            v2 = v2.encode('utf-8')
                        wb['retweet_' + k2] = v2
                else:
                    wb['is_original'] = True
            write_info.append(wb)
        return write_info

    def get_filepath(self, type):
        """获取结果文件路径"""
        try:
            dir_name = self.user['screen_name']
            if self.result_dir_name:
                dir_name = self.user_config['user_id']
            file_dir = os.path.split(os.path.realpath(
                __file__))[0] + os.sep + 'weibo' + os.sep + dir_name
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            file_path = file_dir + os.sep + self.user_config[
                'user_id'] + '.' + type
            return file_path
        except Exception as e:
            logger.exception(e)

    def get_result_headers(self):
        """获取要写入结果文件的表头"""
        result_headers = [
            'id', 'bid', '正文', '头条文章url', '位置', '日期', '工具',
            '点赞数', '评论数', '转发数', '话题', '@用户'
        ]
        if not self.filter:
            result_headers2 = ['是否原创', '源用户id', '源用户昵称']
            result_headers3 = ['源微博' + r for r in result_headers]
            result_headers = result_headers + result_headers2 + result_headers3
        return result_headers

    def write_csv(self, wrote_count):
        """将爬到的信息写入csv文件"""
        write_info = self.get_write_info(wrote_count)
        result_headers = self.get_result_headers()
        result_data = [w.values() for w in write_info]
        file_path = self.get_filepath('csv')
        self.csv_helper(result_headers, result_data, file_path)

    def csv_helper(self, headers, result_data, file_path):
        """将指定信息写入csv文件"""
        if not os.path.isfile(file_path):
            is_first_write = 1
        else:
            is_first_write = 0
        if sys.version < '3':  # python2.x
            with open(file_path, 'ab') as f:
                f.write(codecs.BOM_UTF8)
                writer = csv.writer(f)
                if is_first_write:
                    writer.writerows([headers])
                writer.writerows(result_data)
        else:  # python3.x
            with open(file_path, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                if is_first_write:
                    writer.writerows([headers])
                writer.writerows(result_data)
        if headers[0] == 'id':
            logger.info(u'%d条微博写入csv文件完毕,保存路径:', self.got_count)
        else:
            logger.info(u'%s 信息写入csv文件完毕，保存路径:', self.user['screen_name'])
        logger.info(file_path)

    def update_json_data(self, data, weibo_info):
        """更新要写入json结果文件中的数据，已经存在于json中的信息更新为最新值，不存在的信息添加到data中"""
        data['user'] = self.user
        if data.get('weibo'):
            is_new = 1  # 待写入微博是否全部为新微博，即待写入微博与json中的数据不重复
            for old in data['weibo']:
                if weibo_info[-1]['id'] == old['id']:
                    is_new = 0
                    break
            if is_new == 0:
                for new in weibo_info:
                    flag = 1
                    for i, old in enumerate(data['weibo']):
                        if new['id'] == old['id']:
                            data['weibo'][i] = new
                            flag = 0
                            break
                    if flag:
                        data['weibo'].append(new)
            else:
                data['weibo'] += weibo_info
        else:
            data['weibo'] = weibo_info
        return data

    def write_json(self, wrote_count):
        """将爬到的信息写入json文件"""
        data = {}
        path = self.get_filepath('json')
        if os.path.isfile(path):
            with codecs.open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        weibo_info = self.weibo[wrote_count:]
        data = self.update_json_data(data, weibo_info)
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info(u'%d条微博写入json文件完毕,保存路径:', self.got_count)
        logger.info(path)

    def mysql_create(self, connection, sql):
        """创建MySQL数据库或表"""
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
        finally:
            connection.close()

    def mysql_create_database(self, mysql_config, sql):
        """创建MySQL数据库"""
        try:
            import pymysql
        except ImportError:
            logger.warning(
                u'系统中可能没有安装pymysql库，请先运行 pip install pymysql ，再运行程序')
            sys.exit()
        try:
            if self.mysql_config:
                mysql_config = self.mysql_config
            connection = pymysql.connect(**mysql_config)
            self.mysql_create(connection, sql)
        except pymysql.OperationalError:
            logger.warning(u'系统中可能没有安装或正确配置MySQL数据库，请先根据系统环境安装或配置MySQL，再运行程序')
            sys.exit()

    def mysql_create_table(self, mysql_config, sql):
        """创建MySQL表"""
        import pymysql

        if self.mysql_config:
            mysql_config = self.mysql_config
        mysql_config['db'] = 'newsData'
        connection = pymysql.connect(**mysql_config)
        self.mysql_create(connection, sql)

    def mysql_insert(self, mysql_config, table, data_list):
        """向MySQL表插入或更新数据"""
        import pymysql

        if len(data_list) > 0:
            keys = ', '.join(data_list[0].keys())
            values = ', '.join(['%s'] * len(data_list[0]))
            if self.mysql_config:
                mysql_config = self.mysql_config
            mysql_config['db'] = 'newsData'
            connection = pymysql.connect(**mysql_config)
            cursor = connection.cursor()
            sql = """INSERT INTO {table}({keys}) VALUES ({values}) ON
                     DUPLICATE KEY UPDATE""".format(table=table,
                                                    keys=keys,
                                                    values=values)
            update = ','.join([
                ' {key} = values({key})'.format(key=key)
                for key in data_list[0]
            ])
            sql += update
            try:
                cursor.executemany(
                    sql, [tuple(data.values()) for data in data_list])
                connection.commit()
            except Exception as e:
                connection.rollback()
                logger.exception(e)
            finally:
                connection.close()

    def weibo_to_mysql(self, wrote_count):
        """将爬取的微博信息写入MySQL数据库"""
        mysql_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'password',
            'charset': 'utf8mb4'
        }
        # 创建'weibo'表
        create_table = """
                CREATE TABLE IF NOT EXISTS weibo_news_info (
                id varchar(20) NOT NULL,
                bid varchar(12) NOT NULL,
                user_id varchar(20),
                screen_name varchar(30),
                text varchar(2000),
                article_url varchar(100),
                topics varchar(200),
                at_users varchar(1000),
                location varchar(100),
                created_at DATETIME,
                source varchar(30),
                attitudes_count INT,
                comments_count INT,
                reposts_count INT,
                retweet_id varchar(20),
                PRIMARY KEY (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
        self.mysql_create_table(mysql_config, create_table)
        weibo_list = []
        retweet_list = []
        if len(self.write_mode) > 1:
            info_list = copy.deepcopy(self.weibo[wrote_count:])
        else:
            info_list = self.weibo[wrote_count:]
        for w in info_list:
            if 'retweet' in w:
                w['retweet']['retweet_id'] = ''
                retweet_list.append(w['retweet'])
                w['retweet_id'] = w['retweet']['id']
                del w['retweet']
            else:
                w['retweet_id'] = ''
            weibo_list.append(w)
        # 在'weibo'表中插入或更新微博数据
        self.mysql_insert(mysql_config, 'weibo_news_info', retweet_list)
        self.mysql_insert(mysql_config, 'weibo_news_info', weibo_list)
        logger.info(u'%d条微博写入MySQL数据库完毕', self.got_count)


    def update_user_config_file(self, user_config_file_path):
        """更新用户配置文件"""
        with open(user_config_file_path, 'rb') as f:
            try:
                lines = f.read().splitlines()
                lines = [line.decode('utf-8-sig') for line in lines]
            except UnicodeDecodeError:
                logger.error(u'%s文件应为utf-8编码，请先将文件编码转为utf-8再运行程序',
                             user_config_file_path)
                sys.exit()
            for i, line in enumerate(lines):
                info = line.split(' ')
                if len(info) > 0 and info[0].isdigit():
                    if self.user_config['user_id'] == info[0]:
                        if len(info) == 1:
                            info.append(self.user['screen_name'])
                            info.append(self.start_date)
                        if len(info) == 2:
                            info.append(self.start_date)
                        if len(info) > 2:
                            info[2] = self.start_date
                        lines[i] = ' '.join(info)
                        break
        with codecs.open(user_config_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def write_data(self, wrote_count):
        """将爬到的信息写入文件或数据库"""
        if self.got_count > wrote_count:
            if 'csv' in self.write_mode:
                self.write_csv(wrote_count)
            if 'json' in self.write_mode:
                self.write_json(wrote_count)
            if 'mysql' in self.write_mode:
                self.weibo_to_mysql(wrote_count)
          

    def get_pages(self):
        """获取全部微博"""
        try:
            self.get_user_info()
            self.print_user_info()
            since_date = datetime.strptime(self.user_config['since_date'],
                                           '%Y-%m-%d')
            today = datetime.strptime(str(date.today()), '%Y-%m-%d')
            if since_date <= today:
                page_count = self.get_page_count()
                wrote_count = 0
                page1 = 0
                random_pages = random.randint(1, 5)
                self.start_date = datetime.now().strftime('%Y-%m-%d')
                pages = range(self.start_page, page_count + 1)
                for page in tqdm(pages, desc='Progress'):
                    is_end = self.get_one_page(page)
                    if is_end:
                        break

                    if page % 20 == 0:  # 每爬20页写入一次文件
                        self.write_data(wrote_count)
                        wrote_count = self.got_count

                    # 通过加入随机等待避免被限制。爬虫速度过快容易被系统限制(一段时间后限
                    # 制会自动解除)，加入随机等待模拟人的操作，可降低被系统限制的风险。默
                    # 认是每爬取1到5页随机等待6到10秒，如果仍然被限，可适当增加sleep时间
                    if (page -
                            page1) % random_pages == 0 and page < page_count:
                        sleep(random.randint(6, 10))
                        page1 = page
                        random_pages = random.randint(1, 5)

                self.write_data(wrote_count)  # 将剩余不足20页的微博写入文件
            logger.info(u'微博爬取完成，共爬取%d条微博', self.got_count)
        except Exception as e:
            logger.exception(e)

    def get_user_config_list(self, file_path):
        """获取文件中的微博id信息"""
        with open(file_path, 'rb') as f:
            try:
                lines = f.read().splitlines()
                lines = [line.decode('utf-8-sig') for line in lines]
            except UnicodeDecodeError:
                logger.error(u'%s文件应为utf-8编码，请先将文件编码转为utf-8再运行程序', file_path)
                sys.exit()
            user_config_list = []
            for line in lines:
                info = line.split(' ')
                if len(info) > 0 and info[0].isdigit():
                    user_config = {}
                    user_config['user_id'] = info[0]
                    if len(info) > 2:
                        if self.is_date(info[2]):
                            user_config['since_date'] = info[2]
                        elif info[2].isdigit():
                            since_date = date.today() - timedelta(int(info[2]))
                            user_config['since_date'] = str(since_date)
                    else:
                        user_config['since_date'] = self.since_date
                    if len(info) > 3:
                        user_config['query_list'] = info[3].split(',')
                    else:
                        user_config['query_list'] = self.query_list
                    if user_config not in user_config_list:
                        user_config_list.append(user_config)
        return user_config_list

    def initialize_info(self, user_config):
        """初始化爬虫信息"""
        self.weibo = []
        self.user = {}
        self.user_config = user_config
        self.got_count = 0
        self.weibo_id_list = []

    def start(self):
        """运行爬虫"""
        try:
            for user_config in self.user_config_list:
                if len(user_config['query_list']):
                    for query in user_config['query_list']:
                        self.query = query
                        self.initialize_info(user_config)
                        self.get_pages()
                else:
                    self.initialize_info(user_config)
                    self.get_pages()
                logger.info(u'信息抓取完毕')
                logger.info('*' * 100)
                if self.user_config_file_path and self.user:
                    self.update_user_config_file(self.user_config_file_path)
        except Exception as e:
            logger.exception(e)


def get_config():
    """获取config.json文件信息"""
    config_path = os.path.split(
        os.path.realpath(__file__))[0] + os.sep + 'config.json'
    if not os.path.isfile(config_path):
        logger.warning(u'当前路径：%s 不存在配置文件config.json',
                       (os.path.split(os.path.realpath(__file__))[0] + os.sep))
        sys.exit()
    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.loads(f.read())
            return config
    except ValueError:
        logger.error(u'config.json 格式不正确，请参考 '
                     u'https://github.com/dataabc/weibo-crawler#3程序设置')
        sys.exit()


def main():
    try:
        config = get_config()
        wb = Weibo(config)
        wb.start()  # 爬取微博信息
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    main()
