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
from libact.query_strategies import UncertaintySampling, RandomSampling
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
        # dataset_train = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), localdir)
        # dataset_train = zipfile.ZipFile(docfile).read(localdir).decode('utf-8')
        dataset_train = localdir
        x, y = import_libsvm_sparse(dataset_train).format_sklearn()
        X_train, X_test, label_train, label_test = train_test_split(x, y, test_size=test_size)
        #train_ds = Dataset(X_train,label_train)
        test_ds = Dataset(X_test, label_test)

        return X_train,label_train,test_ds

    def split_train_test_unlabel(self, train, trainlabel, unlabeltext):
        # dataset_unlabel = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), unlabeltext)
        # dataset_unlabel = zipfile.ZipFile(docfile).read(unlabeltext).decode('utf-8')
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

    def sendfile(self, filedir,filetype,username,numneedtobemarked):
        SIZE = 1024
        jsontosend = {}
        jsontosend['filedir'] = filedir
        jsontosend['filetype'] = 1 # 1 means text, 2 means image
        jsontosend['creator'] = username
        jsontosend['numbersneedtobemarked'] = numneedtobemarked
        jsontosend['createtime'] = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        js = json.dumps(jsontosend, sort_keys=True, indent=4, separators=(',', ':'))

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 建立连接:
        s.connect(('47.93.198.34', 8080))
        # 接收欢迎消息:
        print s.recv(SIZE)

        s.send(js)
        print 'json send successfully'
        time.sleep(0.5)
        # print 'command test begins ...'
        # s.send('c')
        # s.send('weeding')
        # print 'command test ended'
        # time.sleep(0.5)

        # print 'image test begins ...'
        # s.send('f')
        # time.sleep(0.2)
        with open(filedir, 'rb') as f:
            for data in f:
                s.send(data)
        print 'file send successfully'

        s.close()
        print 'connection closed'

    def maintodo(self, trainentity, unlabelentity, testentity, unlabeldatasetdir, quota, trainAndtest, testsize, pushallask,docfile,username):
        zipfile.ZipFile(docfile).extractall('/app/codalab/thirdpart/'+username)
        trainentity = '/app/codalab/thirdpart/'+username+'/'+ trainentity
        unlabelentity = '/app/codalab/thirdpart/'+username+'/'+ unlabelentity
        testentity = '/app/codalab/thirdpart/'+username+'/'+ testentity
        #[Todo]:需要标记的个数,是否提交的是训练集+测试集，如果只提交一个测试集，那么划分训练集的比例，
        # quota = 5  
        #1是提交的两个文件，一个训练集一个测试集。
        # trainAndtest = 1
        # testsize = 0.33
        # 1是一次提供所有askid, 0是交互式问询
        # pushallask = 1
        #如果是交互式问询的话，那么需要标记的标记列表
        #interlabel = ['0','1']
        askidlist = []

        
        unlabeldatasetdir = '/app/codalab/thirdpart/'+username+'/unlabel'

        unlabeldatasetdir = os.listdir(unlabeldatasetdir)
        p = open('/app/codalab/thirdpart/test1.txt','w')
        # E_out1, E_out2 = [], []
        # traintest,testtest,ytest,fullytest = split_train_test_origin('99.txt',test_size = 0.33, n_labeled=5)
        unlabeldict = {}

        #[Todo]:提交的是Train还是Train+Test
        if trainAndtest == 1:
            # print trainentity
            # print unlabelentity
            trn_ds,numoftrain,unlabelnames,real_trn_ds = self.split_train_and_unlabel(trainentity,unlabelentity)
            tst_ds = self.split_onlytest(testentity)
        else:
            #[Todo]:提交的只有一个Train
            trn, trn_label, tst_ds = self.split_train_test(trainentity,testsize)
            trn_ds,numoftrain,unlabelnames = self.split_train_test_unlabel(trn, trn_label, unlabelentity)


        # trn_ds2 = copy.deepcopy(trn_ds)


        qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
        #qs2 = RandomSampling(trn_ds2)

        model = LogisticRegression()


        # model.train(trn_ds)
        # E_out1 = np.append(E_out1, 1 - model.score(tst_ds))
        # model.train(trn_ds2)
        # E_out2 = np.append(E_out2, 1 - model.score(tst_ds))

        # query_num = np.arange(0, 1)
        # p1, = ax.plot(query_num, E_out1, 'g', label='qs Eout')
        # p2, = ax.plot(query_num, E_out2, 'k', label='random Eout')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
        #            shadow=True, ncol=5)
        # plt.show(block=False)


        # print ("Label 0 for Ham, Label 1 for Spam")
        # interlabel = ['0','1']
        # lbr = InteractiveLabeler(label_name = interlabel)
        # 
        
        lbr = IdealLabeler(real_trn_ds)#TODO: trn_ds+test_ds to improve


        
        for i in range(quota):
            ask_id = qs.make_query()
            filename = unlabelnames[ask_id-numoftrain]
            if filename.split('/')[-1] in unlabeldatasetdir:
                #filebody = zipfile.ZipFile(docfile).read(traintext).decode('utf-8')
                #unlabeldict[filename] = filebody
                filenamefull = '/app/codalab/thirdpart/'+username+'/unlabel/'+filename.split('/')[-1]
                with open(filenamefull) as f:
                    filebody = f.read()
                    unlabeldict[filenamefull] = filebody
            # print trn_ds.data
            # print trn_ds.data[ask_id]
            X,i = zip(*trn_ds.data)
            # lb = lbr.label(trn_ds.data[ask_id][0])
            # print len(i)

        
            lb = lbr.label(X[ask_id-numoftrain])
            trn_ds.update(ask_id, lb)
            model.train(trn_ds)
            # with open(filename) as f:
            #     print f.read()
            
        csvdir = '/app/codalab/thirdpart/'+username+'/dict.csv'
        with open(csvdir, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in unlabeldict.items():
                askidlist.append(key)
                writer.writerow([key, value])
        #(filedir,filetype,username,numneedtobemarked)
        #self.sendfile(csvdir,1,username,quota)

        # for keys,values in unlabeldict.items():
        #     p.writelines(keys+'\n')
        #     p.writelines(values+'\n')
        return askidlist




