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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import zipfile
import time
import random
import socket
import csv
import json
import base64
import heapq
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
# from makesvm import CreateSVM
from libact.base.dataset import Dataset
from libact.models import LogisticRegression,SVM
from libact.query_strategies import UncertaintySampling, RandomSampling, QueryByCommittee
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.labelers import InteractiveLabeler
from libact.labelers import IdealLabeler
from query_by_committee_plus import QueryByCommitteePlus
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

    def sendfile(self, filedir,filetype,username,useremail,password,numneedtobemarked):
        SIZE = 65535
        RECSIZE = 1024
        jsontosend = {}
        id = time.strftime("%Y", time.localtime())+time.strftime("%m", time.localtime())+time.strftime("%d", time.localtime())+time.strftime("%H", time.localtime())+time.strftime("%M", time.localtime())+time.strftime("%S", time.localtime())
        id = id + str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))+str(random.randint(0,9))
        jsontosend['id'] = long(id)
        jsontosend['username'] = username
        jsontosend['email'] = useremail
        jsontosend['password'] = password
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

    def encryptbmt(self, bmtpassword):
        pwd = bmtpassword[::-1]
        k = 0
        newpwd = ''

        for i in pwd:
            i = chr(ord(i)-k)
            k = k + 1
            newpwd = newpwd + i

        newpwd2 = base64.b64encode(newpwd)

        newpwd3 = 'T' + newpwd2 + 'B'

        newpwd4=''
        for i in newpwd3:
            i = chr(ord(i)+1)
            newpwd4 = newpwd4 + i

        newpassword = base64.b64encode(newpwd4)

        return newpassword


    def split_for_drawplot(self, trn_ds, numoftrain, quota):
        #n_labeled = numoftrain - quota

        #if n_labeled < 10:
        #    n_labeled = 1
        #else:
        #    n_labeled = 5

        n_labeled = 0
        flag = 0
        #n_labeled = numoftrain - quota
        #n_labeled = int(n_labeled/6)
        #if n_labeled < 1:
        #    n_labeled = 1



        x_train,y_train = trn_ds.format_sklearn()[0],trn_ds.format_sklearn()[1]

        try:
            if numoftrain >=1000:
                n_labeled = int(numoftrain/40)
            elif numoftrain >=500:
                n_labeled = int(numoftrain/20)
            elif numoftrain >=100:
                n_labeled = int(numoftrain/10)
            else:
                n_labeled = 6
        except:
            for i in range(len(y_train)):
                if y_train[i] != y_train[i+1]:
                    flag=1
                    n_labeled = i+2
                    break
                flag = -1
        if flag ==-1:
            pass#[todo]报错


        none_trn_ds = Dataset(x_train, np.concatenate(
            [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

        return none_trn_ds


    #空跑测试集做图像
    def score_ideal(self, trn_ds,tst_ds,lbr,model,qs,quota):
        E_in, E_out = [], []

        for _ in range(quota):
            ask_id = qs.make_query()

            X, _ = zip(*trn_ds.data)
            c = len(X)
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)
            model.train(trn_ds)
            E_in = np.append(E_in, 1 - model.score(trn_ds))#in-sample error
            E_out = np.append(E_out, 1 - model.score(tst_ds))#out-sample error
        return E_in, E_out

    def plotforimage(self, query_num,E_in1,E_in2,E_out1,E_out2,username):
        dir = '/app/codalab/static/img/partpicture/'+username+'/'

        if os.path.isfile(dir + 'compare.png'):
            os.remove(dir + 'compare.png')
        plt.plot(query_num, E_in1, 'b', label='qs Ein')
        plt.plot(query_num, E_in2, 'r', label='random Ein')
        plt.plot(query_num, E_out1, 'g', label='qs Eout')
        plt.plot(query_num, E_out2, 'k', label='random Eout')
        plt.xlabel('Number of Queries')
        plt.ylabel('Error')
        plt.title('Experiment Result')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)

        isExists = os.path.exists(dir)
        if not isExists:
            os.makedirs(dir)
        plt.savefig(dir + 'compare.png')
        plt.clf()
        plt.close()




    def maintodo(self, kind, modelselect, strategy, algorithm, quota, trainAndtest, testsize, pushallask,docfile,username,useremail,bmtpassword):
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
        E_in1, E_in2 = [], []
        E_out1, E_out2 = [], []
        unlabeldict = {}

        #提交的是Train还是Train+Test
        if trainAndtest == 1:
            trn_ds,numoftrain,unlabelnames,real_trn_ds = self.split_train_and_unlabel(trainentity,unlabelentity)
            tst_ds = self.split_onlytest(testentity)
            none_trn_ds = self.split_for_drawplot(real_trn_ds, numoftrain, quota)
        else:
            #提交的只有一个Train
            trn, trn_label, tst_ds = self.split_train_test(trainentity,testsize)
            trn_ds,numoftrain,unlabelnames = self.split_train_test_unlabel(trn, trn_label, unlabelentity)
            real_trn_ds = copy.deepcopy(trn_ds)
            none_trn_ds = self.split_for_drawplot(real_trn_ds, numoftrain, quota)


        trn_ds_random = copy.deepcopy(none_trn_ds)#原验证集强行划分部分未知None做效果用
        qs_random = RandomSampling(trn_ds_random)

        #Todo:补充多种策略、算法
        if modelselect == 'logic':
            if strategy == 'binary':
                if algorithm == 'qbc':
                    qs = QueryByCommitteePlus(trn_ds,models=[LogisticRegression(C=1.0),LogisticRegression(C=0.4),],)
                    qs_fordraw = QueryByCommittee(none_trn_ds,models=[LogisticRegression(C=1.0),LogisticRegression(C=0.4),],)
                elif algorithm == 'us':
                    qs = UncertaintySampling(trn_ds,model=SVM(decision_function_shape='ovr'))
                    qs_fordraw = UncertaintySampling(none_trn_ds, method='lc', model=LogisticRegression())
                else:
                    pass
                model = LogisticRegression()
            elif strategy == 'multiclass':#[todo]need to be improved
                if algorithm == 'qbc':
                    qs = QueryByCommitteePlus(trn_ds,models=[LogisticRegression(C=1.0),LogisticRegression(C=0.4),],)
                    qs_fordraw = QueryByCommittee(none_trn_ds,models=[LogisticRegression(C=1.0),LogisticRegression(C=0.4),],)
                elif algorithm == 'us':
                    qs = UncertaintySampling(trn_ds,model=SVM(decision_function_shape='ovr'))
                    qs_fordraw = UncertaintySampling(none_trn_ds, method='lc', model=LogisticRegression())
                else:
                    pass
                model = LogisticRegression()

            else: #multilabel
                pass

            lbr = IdealLabeler(real_trn_ds)

            E_in1, E_out1 = self.score_ideal(none_trn_ds,tst_ds,lbr,model,qs_fordraw,quota)
            model = LogisticRegression()

            E_in2, E_out2 = self.score_ideal(trn_ds_random,tst_ds,lbr,model,qs_random,quota)



        elif modelselect == 'svm':
            if strategy == 'binary':
                if algorithm == 'qbc':
                    qs = QueryByCommitteePlus(trn_ds,models=[SVM(C=1.0, decision_function_shape='ovr'),SVM(C=0.4, decision_function_shape='ovr'),],)
                    qs_fordraw = QueryByCommittee(none_trn_ds,models=[SVM(C=1.0, decision_function_shape='ovr'),SVM(C=0.4, decision_function_shape='ovr'),],)
                elif algorithm == 'us':
                    qs = UncertaintySampling(trn_ds, model=SVM(decision_function_shape='ovr'))
                    qs_fordraw = UncertaintySampling(none_trn_ds, model=SVM(decision_function_shape='ovr'))
                else:
                    pass
                model = SVM(kernel='linear', decision_function_shape='ovr')
            elif strategy == 'multiclass':
                pass

            else: #multilabel
                pass
            lbr = IdealLabeler(real_trn_ds)

            E_in1, E_out1 = self.score_ideal(none_trn_ds,tst_ds,lbr,model,qs_fordraw,quota)
            model = SVM(kernel='linear', decision_function_shape='ovr')
            E_in2, E_out2 = self.score_ideal(trn_ds_random,tst_ds,lbr,model,qs_random,quota)
        else:
            pass

        
#        lbr = IdealLabeler(real_trn_ds)

#        E_in1, E_out1 = self.score_ideal(none_trn_ds,tst_ds,lbr,model,qs_fordraw,quota)
#        model = LogisticRegression()

#        E_in2, E_out2 = self.score_ideal(trn_ds_random,tst_ds,lbr,model,qs_random,quota)

        self.plotforimage(np.arange(1,quota+1),E_in1,E_in2,E_out1,E_out2,username)


#dir = '/app/codalab/static/img/partpicture/'+username+'/'
        #返回一批实例,返回分数是为了解决不标注的情况下无法自动更新的问题
        if pushallask == 1:

            first, scores = qs.make_query(return_score = True)
            number, num_score = zip(*scores)[0], zip(*scores)[1]
            num_score_array = np.array(num_score)
            max_n = heapq.nlargest(quota,range(len(num_score_array)),num_score_array.take)

            if len(unlabeldatasetdir) < 1:
                for ask_id in max_n:
                    filename = unlabelnames[ask_id]
                    askidlist.append(filename)

                csvdir = '/app/codalab/static/img/partpicture/'+username+'/dict.csv'
                with open(csvdir, 'wb') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['name'])
                    for unlabelname in askidlist:
                        writer.writerow([unlabelname])

                return askidlist,csvdir
            else:
                for ask_id in max_n:
                    filename = unlabelnames[ask_id]
                    if filename.split('/')[-1] in unlabeldatasetdir:
                        filenamefull = '/app/codalab/thirdpart/'+username+'/unlabel/'+filename.split('/')[-1]
                        with open(filenamefull) as f:
                            filebody = f.read()
                            unlabeldict[filename] = filebody

                    askidlist.append(filename)

                csvdir = '/app/codalab/static/img/partpicture/'+username+'/dict.csv'
                with open(csvdir, 'wb') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['name','entity'])
                    for key, value in unlabeldict.items():
                        #askidlist.append(key)
                        writer.writerow([key, value])
                return askidlist,csvdir


        #向标注平台发送
        else:
            first, scores = qs.make_query(return_score = True)
            number, num_score = zip(*scores)[0], zip(*scores)[1]
            num_score_array = np.array(num_score)
            max_n = heapq.nlargest(quota,range(len(num_score_array)),num_score_array.take)


            for ask_id in max_n:
                filename = unlabelnames[ask_id]
                if filename.split('/')[-1] in unlabeldatasetdir:
                    filenamefull = '/app/codalab/thirdpart/'+username+'/unlabel/'+filename.split('/')[-1]
                    with open(filenamefull) as f:
                        filebody = f.read()
                        unlabeldict[filename] = filebody

                askidlist.append(filename)

            csvdir = '/app/codalab/thirdpart/'+username+'/dict.csv'
            with open(csvdir, 'wb') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['name','entity'])
                for key, value in unlabeldict.items():
                    #askidlist.append(key)
                    writer.writerow([key, value])
            newpassword = self.encryptbmt(bmtpassword)

            #rec_status, rec_url = self.sendfile(csvdir,11,username,useremail,newpassword,quota)

            rec_status = 0
            rec_url = 'www.baidu.com'
            return rec_status, rec_url, askidlist
