#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# file: config.py
# author: jiangqr
# data: 2017.1.3
# note: receive and send message for web
#

import threading
import os


#class MysqlConfig:
#    def __init__(self, host, port, user, password, database):
#        self.host = host
#        self.port = port
#        self.user = user
#        self.password = password
#        self.database = database
#        self.data_sheet_recognition = 'recognition2_scene'
#        self.data_sheet_class = 'recognition2_class'
#
## mysql and socket info
#MYSQL_INFO = None
#ZMQ_TCP = None
#ZMQ_THREAD_NUM = 2
#VIDEO_THREAD_NUM = 1
#
#SERVER_VERSION = 'vcatools'
#if SERVER_VERSION == 'dev1':
#    GPU_ID = 0
#    MYSQL_INFO = MysqlConfig(host='test1.bjvca.com', port=3306, user='root', password='18576698510', database='new_vca_b')
#    ZMQ_TCP = 'tcp://*:7701'
#    REMOTE_FTP = 'image.bjvca.com'
#    REMOTE_USER = 'user'
#    REMOTE_PASSWORD = 'vca666'
#    REMOTE_ROOT_PATH = 'imgserver/'
#    REMOTR_RELATIVE_PATH = 'static/dev_identify/'
#elif SERVER_VERSION == 'vcatools':
#    GPU_ID = 1
#    MYSQL_INFO = MysqlConfig(host='vcasltdb.mysql.rds.aliyuncs.com', port=3306, user='dsp_user', password='VcaSlt20171001', database='b_vca')
#    ZMQ_TCP = 'tcp://*:7703'
#    REMOTE_FTP = '223.223.180.16'
#    REMOTE_USER = 'user'
#    REMOTE_PASSWORD = 'VcaSlt20171011'
#    REMOTE_ROOT_PATH = '/'
#    REMOTR_RELATIVE_PATH = 'static/snapshot/'
#    REMOTR_REMOTR_PATH = 'imgserver/'

# model
MODEL_FILE = 'model/inception_resnet_v2_behaviour_337_5_22_411k_split.pb'
LABEL_FILE = 'model/labels_337_4_10.txt'
BOOL_V2_MODEL = 1 if MODEL_FILE.find('v2') >= 0 else 0
OUTPUT_THRESHOLD = 0.5
POST_THRESHOLD = 0.7
CONTINUE_TIME = 1000
FRAME_GAP = 10

mutex = threading.Lock()
