# -*- coding: UTF-8 -*-
'''
'''
import socket, time, os, threading, json
import httplib, urllib

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

class SendtoBMT():
    def __init__(self, username, password):
        self.ip = '10.2.26.114'
        self.port = 7777
        self.username = username
        self.password = password

    def gettoken(self, username, password):
        httpClient = httplib.HTTPConnection(self.ip, self.port, timeout=30)
        body = urllib.urlencode({'username': username, 'password': password})
        header = {
            'Content-type': 'application/x-www-form-urlencoded',
        }

        httpClient.request('POST', '/api-token-auth/', body=body, headers=header)

        response = httpClient.getresponse()
        # print response.status
        # print response.reason
        token = response.read()
        # print token
        f = open('jwt_token.html', 'w+')
        # print >> f, token
        f.close()
        return token


    def useapi(self, token, title, typeno, dataset, describe, price, deadline, rank_required, max_allocation_per_task,
               template_id, config):
        httpClient = httplib.HTTPConnection(self.ip, self.port, timeout=30)

        # 任务信息
        info = {}

        info['operation'] = 'CreateTask'
        # [C].Sting, the title
        info['title'] = title
        # [C].Int, 0-classify, 1-read materials and mark, 2-DIY
        info['type'] = typeno
        # [C].String, json type?
        info['dataset'] = dataset
        # [O]. Worker can read when they open the task
        info['describe'] = describe
        # [C]. Int, worker can get how many grades when they finish one question. 0 is none.
        info['price'] = price
        # [O]. String, '20160518010201', null is no deadline
        info['deadline'] = deadline
        # [O]. String, Hold. the requirement of worker's ability
        info['rank_required'] = rank_required
        # [O]. Int, how many workers required to answer one question. Deault: finite
        info['max_allocation_per_task'] = max_allocation_per_task
        # [O/C]. String, only if type==2 then C, else O. can look up at BMT
        info['template_id'] = template_id
        # [O/C]. String, json type. Only if type==0 then C, else O. {question:'spam or not?', option:'yes.no；1.0'}
        info['config'] = config

        refs = urllib.urlencode(info)

        tokenjs = json.loads(token)

        # print tokenjs['token']
        token = 'JWT ' + tokenjs['token']
        # for key,item in token:
        # print item
        # print token
        header = {
            'Authorization': token
        }

        urls = '/restful/?' + refs

        httpClient.request('GET', urls, headers=header)

        response = httpClient.getresponse()
        # print response.status
        # print response.reason
        data = response.read()

        f = open('createTask.html', 'w+')
        # print >> f, data
        f.close()

        # print 'httpClient closed'
        # print data
        return data, response.status


    def getResult(self, token, taskid):
        # code=1成功
        httpClient = httplib.HTTPConnection(self.ip, self.port, timeout=30)

        info = {}
        info['operation'] = 'GetResult'
        info['task_id'] = taskid

        refs = urllib.urlencode(info)
        tokenjs = json.loads(token)

        print tokenjs['token']
        token = 'JWT ' + tokenjs['token']
        header = {
            'Authorization': token
        }
        urls = '/restful/?' + refs
        httpClient.request('GET', urls, headers=header)

        response = httpClient.getresponse()
        # print response.status
        # print response.reason
        data = response.read()
        # print data

        f = open('getResult.html', 'w+')
        # print >> f, data
        f.close()

        # print 'httpClient closed'/
        return data, response.status


    def decidetosend(self, title, typeno, dataset, describe, price, deadline, question, visual_value, truth_value):
        title = title
        typeno = typeno
        # dataset = dataset
        # dataset = [
        #     ['content'],
        #     ['qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'],
        #     ['wwwwwwwwwwwwwwwwwwwwwwwwwwwww'],
        #     ['eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee']
        # ]
        dataset = json.dumps(dataset)
        describe = describe
        price = price
        deadline = deadline
        # deadline = '20180808080808'
        rank_required = ''
        max_allocation_per_task = 5
        template_id = ''
        config = {
            'question': question,
            'option' : {
                'visual_value':visual_value,
                'truth_value':truth_value
            }
        }
        # config = {
        #     'question': 'IS SPAM?',
        #     'option': {
        #         'visual_value': ['yes', 'no'],
        #         'truth_value': [1, 0]
        #     }
        # }
        config = json.dumps(config)

        token = self.gettoken(self.username, self.password)
        taskdata, status = self.useapi(token, title, typeno, dataset, describe, price, deadline, rank_required, max_allocation_per_task,
               template_id, config)
        # self.getResult(token)
        return taskdata, status

