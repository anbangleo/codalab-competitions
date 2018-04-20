#!/usr/bin/env python3
# coding=utf-8
"""
This script simulates real world use of active learning algorithms. Which in the
start, there are only a small fraction of samples are labeled. During active
learing process active learning algorithm (QueryStrategy) will choose a sample
from unlabeled samples to ask the oracle to give this sample a label (Labeler).

In this example, ther dataset are from the digits dataset from sklearn. User
would have to label each sample choosed by QueryStrategy by hand. Human would
label each selected sample through InteractiveLabeler. Then we will compare the
performance of using UncertaintySampling and RandomSampling under
LogisticRegression.
"""

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import time
import random
import socket
import csv
import json
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
# from makesvm import CreateSVM
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling, QueryByCommittee
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.labelers import InteractiveLabeler
from libact.labelers import IdealLabeler
class BinaryClassTest(object):
    def __init__(self):
        pass
     
    def split_train_and_unlabel(self, traintext, unlabeltext):
        # dataset_train = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), traintext)
        # dataset_unlabel = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), unlabeltext)
        # dataset_train = zipfile.ZipFile(docfile).read(traintext).decode('utf-8')
        # dataset_unlabel = zipfile.ZipFile(docfile).read(unlabeltext).decode('utf-8')

        #todo need to change it to path
        #原训练集
        x_train, label_train = import_libsvm_sparse(traintext).format_sklearn()
        numoftrain = len(x_train)

        #未标记集
        allunlabel = np.loadtxt(unlabeltext, dtype = str)

        #减1减去的是unlabel向量的[名称]
        unlabel_data = np.zeros((np.shape(allunlabel)[0],np.shape(allunlabel)[1]-1))
        unlabel_name = []

        for i in range(np.shape(allunlabel)[0]):
            unlabel_name.append(allunlabel[i][0])
            for j in range(np.shape(allunlabel)[1]-1):
                unlabel_data[i][j] = allunlabel[i][j+1].split(':')[1]

        x = np.vstack((x_train, unlabel_data))
        y = np.hstack((label_train,[None]*len(allunlabel)))

        trn_ds = Dataset(x,y)
        real_trn_ds = Dataset(x_train,label_train)

        #返回训练集，训练集的个数，未标记数据的名称list
        return trn_ds, numoftrain, unlabel_name,real_trn_ds


    def split_onlytest(self, localdir):
        #返回测试集样例
        # dataset_train = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), localdir)
        # dataset_train = zipfile.ZipFile(docfile).read(localdir).decode('utf-8')
        dataset_train = localdir
        x, y = import_libsvm_sparse(dataset_train).format_sklearn()
        test_ds = Dataset(x,y)
        return test_ds


    def split_train_test (self, localdir, test_size):
        dataset_train = localdir
        x, y = import_libsvm_sparse(dataset_train).format_sklearn()
        X_train, X_test, label_train, label_test = train_test_split(x, y, test_size=test_size)
        #train_ds = Dataset(X_train,label_train)
        test_ds = Dataset(X_test, label_test)

        return X_train,label_train,test_ds

    def split_train_test_unlabel(self, train, trainlabel, unlabeltext):

        dataset_unlabel = unlabeltext
        #原训练集
        numoftrain = len(train)

        #未标记集
        allunlabel = np.loadtxt(dataset_unlabel, dtype = str)

        #减1减去的是unlabel向量的[名称]
        unlabel_data = np.zeros((np.shape(allunlabel)[0],np.shape(allunlabel)[1]-1))
        unlabel_name = []

        for i in range(np.shape(allunlabel)[0]):
            unlabel_name.append(allunlabel[i][0])
            for j in range(np.shape(allunlabel)[1]-1):
                unlabel_data[i][j] = allunlabel[i][j+1].split(':')[1]

        x = np.vstack((train, unlabel_data))
        y = np.hstack((trainlabel,[None]*len(allunlabel)))

        trn_ds = Dataset(x,y)

        #返回训练集，训练集的个数，未标记数据的名称list
        return trn_ds, numoftrain, unlabel_name

    def sendfile(self, filedir,filetype,username,useremail,numneedtobemarked):
        SIZE = 65535
        RECSIZE = 1024
        jsontosend = {}
        id = time.strftime("%Y", time.localtime())+time.strftime("%m", time.localtime())+time.strftime("%d", time.localtime())+time.strftime("%H", time.localtime())+time.strftime("%M", time.localtime())+time.strftime("%S", time.localtime())
        id = id + str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))
        jsontosend['id'] = long(id)
        jsontosend['username'] = username
        jsontosend['email'] = useremail
        jsontosend['password'] = 'abcdefg'
        jsontosend['type'] = filetype # 11 means text, 2 means image

        jsontosend['numbersneedtobemarked'] = numneedtobemarked
        jsontosend['createtime'] = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        #js = json.dumps(jsontosend, sort_keys=True, indent=4, separators=(',', ':'))
        js = json.dumps(jsontosend)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('10.2.26.114', 11223))


        s.send('c')
        time.sleep(0.2)


        s.send(js)

        time.sleep(0.5)
        s.send('f')
        time.sleep(0.2)
        with open(filedir, 'rb') as f:
            for data in f:
                s.send(data)

        time.sleep(0.5)

        data = s.recv(RECSIZE)
        rec = json.loads(data)
        rec_status = rec['status']
        rec_url = rec['url']
        s.close()
        return rec_status, rec_url


    def maintodo(self, kind, model, strategy, algorithm, quota, trainAndtest, testsize, pushallask,docfile,username,useremail):
        zipfile.ZipFile(docfile).extractall('/app/codalab/thirdpart/'+username)
        trainentity = '/app/codalab/thirdpart/'+username+'/'+ 'train.txt'
        unlabelentity = '/app/codalab/thirdpart/'+username+'/'+ 'unlabel.txt'
        testentity = '/app/codalab/thirdpart/'+username+'/'+ 'test.txt'
        #[Todo]:需要标记的个数,是否提交的是训练集+测试集，如果只提交一个测试集，那么划分训练集的比例，
        #1是提交的两个文件，一个训练集一个测试集。
        # trainAndtest = 1
        # testsize = 0.33
        # 1是一次提供所有askid, 0是交互式问询
        # pushallask = 1
        #如果是交互式问询的话，那么需要标记的标记列表
        #interlabel = ['0','1']
        askidlist = []

        #Todo:ask_id是问的train+unlabel中unlabel的id

        
        unlabeldatasetdir = '/app/codalab/thirdpart/'+username+'/unlabel'

        unlabeldatasetdir = os.listdir(unlabeldatasetdir)

        # E_out1, E_out2 = [], []
        unlabeldict = {}

        #[Todo]:提交的是Train还是Train+Test
        if trainAndtest == 1:
            trn_ds,numoftrain,unlabelnames,real_trn_ds = self.split_train_and_unlabel(trainentity,unlabelentity)
            tst_ds = self.split_onlytest(testentity)
        else:
            #[Todo]:提交的只有一个Train
            trn, trn_label, tst_ds = self.split_train_test(trainentity,testsize)
            trn_ds,numoftrain,unlabelnames = self.split_train_test_unlabel(trn, trn_label, unlabelentity)
            real_trn_ds = copy.deepcopy(trn_ds)


        # trn_ds2 = copy.deepcopy(trn_ds)

        #qs2 = RandomSampling(trn_ds2)

        #Todo:补充多种策略、算法
        if strategy == 'binary':
            if algorithm == 'qbc':
                qs = QueryByCommittee(trn_ds,models=[LogisticRegression(C=1.0),LogisticRegression(C=0.1),],)
            elif algorithm == 'us':
                qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
            else:
                pass
            model = LogisticRegression()
        elif strategy == 'multiclass':
            pass

        else: #multilabel
            pass


        
        lbr = IdealLabeler(real_trn_ds)#TODO: trn_ds+test_ds to improve

        if pushallask == 1:
            for i in range(quota):
                ask_id = qs.make_query()
                filename = unlabelnames[ask_id-numoftrain]

                X,i = zip(*trn_ds.data)

                lb = lbr.label(X[ask_id-numoftrain])
                trn_ds.update(ask_id, lb)
                model.train(trn_ds)
                askidlist.append(filename)
            return askidlist

        #向标注平台发送
        else:
            for i in range(quota):
                ask_id = qs.make_query()
                filename = unlabelnames[ask_id-numoftrain]
                if filename.split('/')[-1] in unlabeldatasetdir:
                    #filebody = zipfile.ZipFile(docfile).read(traintext).decode('utf-8')
                    #unlabeldict[filename] = filebody
                    filenamefull = '/app/codalab/thirdpart/'+username+'/unlabel/'+filename.split('/')[-1]
                    with open(filenamefull) as f:
                        filebody = f.read()
                        unlabeldict[filename] = filebody

                X,i = zip(*trn_ds.data)

                lb = lbr.label(X[ask_id-numoftrain])
                trn_ds.update(ask_id, lb)
                model.train(trn_ds)
                askidlist.append(filename)

            csvdir = '/app/codalab/thirdpart/'+username+'/dict.csv'
            with open(csvdir, 'wb') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['name','entity'])
                for key, value in unlabeldict.items():
                    #askidlist.append(key)
                    writer.writerow([key, value])
            rec_status, rec_url = self.sendfile(csvdir,11,username,useremail,quota)
            #rec_status = 0
            #rec_url = ''
            return rec_status, rec_url, askidlist



