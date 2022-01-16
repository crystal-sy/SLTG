# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2 20:00:12 2021

@author: styra
"""
import csv
import json
import os
import random
import sys
from collections import OrderedDict
from datetime import date, datetime, timedelta
from time import sleep

import requests
from lxml import etree
 
# 项目路径,将项目路径保存
project_path = 'D:\sycode\SLTG\src\main'
sys.path.append(project_path)

from config import sltg_config as sltg_config
from spider import newsSpiderDb as db

import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")
    
logging.config.fileConfig(sltg_config.logging_path)
logger = logging.getLogger('spider')

class Weibo(object):
    def __init__(self, config):
        """Weibo类初始化"""
        self.validate_config(config)
        self.filter = config['filter']  # 取值范围为0、1,程序默认值为0,代表要爬取用户的全部微博,1代表只爬取用户的原创微博
        self.remove_html_tag = config['remove_html_tag']  # 取值范围为0、1, 0代表不移除微博中的html tag, 1代表移除
        since_date = config['since_date']
        if isinstance(since_date, int):
            since_date = date.today() - timedelta(since_date)
        
        since_date = str(since_date)
        self.since_date = since_date  # 起始时间，即爬取发布日期从该值到现在的微博，形式为yyyy-mm-dd
        self.start_page = config.get('start_page', 1)  # 开始爬的页，如果中途被限制而结束可以用此定义开始页码
        self.result_dir_name = config.get('result_dir_name', 0)  # 结果目录名，取值为0或1，决定结果文件存储在用户昵称文件夹里还是用户id文件夹里
        cookie = config.get('cookie')  # 微博cookie，可填可不填
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
        self.headers = {'User_Agent': user_agent, 'Cookie': cookie}
        
        user_id_list = config['user_id_list']
        user_config_list = [{
            'user_id': user_id,
            'since_date': self.since_date
        } for user_id in user_id_list]
        
        self.user_config_list = user_config_list  # 要爬取的微博用户的user_config列表
        self.user_config = {}  # 用户配置,包含用户id和since_date
        self.user = {}  # 存储目标微博用户信息
        self.got_count = 0  # 存储爬取到的微博数
        self.weibo_id_list = []  # 存储爬取到的所有微博id
        self.go_on = True

    def validate_config(self, config):
        """验证配置是否正确"""
        # 验证filter
        argument_list = ['filter']
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
            
        if since_date > sltg_config.now_date:
            logger.warning(u'since_date值不能大于当前日期,请重新输入')
            sys.exit()

        # 验证user_id_list
        user_id_list = config['user_id_list']
        if (not isinstance(user_id_list, list)) and (not user_id_list.endswith('.txt')):
            logger.warning(u'user_id_list值应为list类型或txt文件路径')
            sys.exit()
        if not isinstance(user_id_list, list):
            if not os.path.isabs(user_id_list):
                user_id_list = os.path.split(os.path.realpath(__file__))[0] + os.sep + user_id_list
            if not os.path.isfile(user_id_list):
                logger.warning(u'不存在%s文件', user_id_list)
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
            'containerid': '107603' + str(self.user_config['user_id']),
            'page': page
        }
        js = self.get_json(params)
        return js

    def user_to_csv(self):
        """将爬取到的用户信息写入csv文件"""
        file_dir = sltg_config.weibo_spider
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        result_headers = [
            '用户id', '昵称', 
            '微博数', '粉丝数', '关注数', 
            '是否认证', '认证类型', '认证信息', 'url'
        ]
        result_data = [[
            v.encode('utf-8') if 'unicode' in str(type(v)) else v
            for v in self.user.values()
        ]]
        self.csv_helper(result_headers, result_data, sltg_config.weibo_user_path)

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
            user_info['url'] = sltg_config.weibo_profile_url + self.user_config['user_id']
            user = self.standardize_info(user_info)
            self.user = user
            self.user_to_csv()
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
        weibo['reposts_count'] = self.string_to_int(
            weibo_info.get('reposts_count', 0))
        weibo['topics'] = self.get_topics(selector)
        weibo['at_users'] = self.get_at_users(selector)
        return self.standardize_info(weibo)

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
                for w in weibos:
                    if w['card_type'] == 9 and self.go_on :
                        wb = self.get_one_weibo(w)
                        if wb:
                            if wb['id'] in self.weibo_id_list:
                                continue
                                
                            created_at = datetime.strptime(
                                wb['created_at'], '%Y-%m-%d')
                            since_date = datetime.strptime(
                                self.user_config['since_date'], '%Y-%m-%d')
                            if created_at != since_date:
                                self.go_on = False
                                break
                                
                            if self.is_pinned_weibo(w):
                                continue
                            else:
                                self.weibo_to_mysql(wb)
                                
                            if 'retweet' not in wb.keys() :
                                self.weibo_id_list.append(wb['id'])
                                self.got_count += 1
                            else:
                                logger.info(u'正在过滤转发微博')
            logger.info(u'{}已获取{}({})的第{}页微博{}'.format(
                '-' * 30, self.user['screen_name'], self.user['id'], page,
                '-' * 30))
        except Exception as e:
            logger.exception(e)

    def csv_helper(self, headers, result_data, file_path):
        """将指定信息写入csv文件"""
        if not os.path.isfile(file_path):
            is_first_write = 1
        else:
            is_first_write = 0
            
        with open(file_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if is_first_write:
                writer.writerows([headers])
            writer.writerows(result_data)
            
        logger.info(u'%s 信息写入csv文件完毕，保存路径:', self.user['screen_name'])
        logger.info(file_path)

    def weibo_to_mysql(self, wb):
        """将爬取的微博信息写入MySQL数据库"""
        for k, v in wb.items():
            if k not in ['user_id', 'screen_name', 'retweet']:
                if 'unicode' in str(type(v)):
                    v = v.encode('utf-8')
                wb[k] = v
        wb['spider'] = 'weibo_' + wb['screen_name']
        wb['url'] = sltg_config.weibo_article_url + str(wb['id'])
        db.insert_news_info_weibo(wb)

    def get_pages(self):
        """获取全部微博"""
        try:
            self.get_user_info()
            random_pages = random.randint(1, 10)
            page = self.start_page
            while self.go_on and self.got_count < 20:
                self.get_one_page(page)
                page += 1

                # 通过加入随机等待避免被限制。爬虫速度过快容易被系统限制(一段时间后限
                # 制会自动解除)，加入随机等待模拟人的操作，可降低被系统限制的风险。默
                # 认是每爬取1到5页随机等待6到10秒，如果仍然被限，可适当增加sleep时间
                if self.go_on and page % random_pages == 0 :
                    sleep(random.randint(6, 10))
                    
            logger.info(u'微博爬取完成，共爬取%d条微博', self.got_count)
        except Exception as e:
            logger.exception(e)

    def initialize_info(self, user_config):
        """初始化爬虫信息"""
        self.user = {}
        self.user_config = user_config
        self.got_count = 0
        self.weibo_id_list = []

    def start(self):
        """运行爬虫"""
        try:
            for user_config in self.user_config_list:
                self.initialize_info(user_config)
                self.get_pages()
                logger.info(u'信息抓取完毕')
                logger.info('*' * 100)
        except Exception as e:
            logger.exception(e)

def get_config():
    """获取config.json文件信息"""
    config_path = sltg_config.weibo_config
    if not os.path.isfile(config_path):
        logger.warning(u'当前路径：%s 不存在配置文件config.json', config_path)
        sys.exit()
    try:
        with open(config_path, encoding='utf-8') as f:
            return json.loads(f.read())
    except ValueError:
        logger.error(u'config.json文件格式不正确')
        sys.exit()

def main(sinceDate):
    try:
        user_csv_file = sltg_config.weibo_user_path
        if os.path.exists(user_csv_file):
            os.remove(user_csv_file)
            
        weibo_config = get_config()
        weibo_config['since_date'] = sinceDate
        wb = Weibo(weibo_config)
        wb.start()  # 爬取微博信息
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    sinceDate = sys.argv[1]
    # sinceDate = '2022-01-04'
    main(sinceDate)
