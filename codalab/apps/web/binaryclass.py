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
import sys
sys.path.append('/app/codalab/apps/web/deeplearning')
import copy
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import zipfile
import time
import random
import socket
import csv
import json
import base64
import heapq
matplotlib.use('Agg')
# libact classes
# from makesvm import CreateSVM
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling, RandomSampling, QueryByCommittee, ActiveLearningByLearning
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.labelers import InteractiveLabeler
from libact.labelers import IdealLabeler
from query_by_committee_plus import QueryByCommitteePlus
from active_learning_by_learning_plus import ActiveLearningByLearningPlus
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
import tensorflow.contrib.keras as kr
# from cp-cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

from dealwordindict import read_vocab, read_category, batch_iter, process_file, process_file_rnn, build_vocab, native_content
import time
from datetime import timedelta
import heapq
from rnnmodel import RNN_Probability_Model, TRNNConfig
import random


class BinaryClassTest(object):
    def __init__(self):
        pass

    def get_time_dif(start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    # 将多维特征矩阵映射到一维
    def convertlabel(self, label_id):
        resultlist = []
        for i in range(np.shape(label_id)[0]):
            for j in range(np.shape(label_id)[1]):
                if label_id[i][j] == 1:
                    resultlist.append(j)
        resultlist = np.array(resultlist)
        return resultlist

    # 提交的是Train和Test，此时只需要处理Train即可，将Train和Unlabel组合
    def split_train_and_unlabel(self, train_dir, unlabel_dir, test_dir, vocab_dir, vocab_size):

        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)

        x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, wordslength)
        listy = []
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if y[i][j] == 1:
                    listy.append(j)
        listy = np.array(listy)

        x_train, x_test, y_train, y_test = \
            train_test_split(x_train, listy, test_size=1)
        # todo need to change it to path

        # 原训练集
        # x_intrain, label_train = import_libsvm_sparse(traintext).format_sklearn()
        # numoftrain = len(x_train)

        #未标记集


        allunlabel = np.loadtxt(unlabeltext, dtype = str)

        #减1减去的是unlabel向量的[名称]
            # unlabel_data = np.zeros((np.shape(allunlabel)[0],np.shape(allunlabel)[1]-1))
        unlabel_name = []
        # x_unlabel and y_unlabel is np.array
        x_unlabel, y_unlabel = process_file(unlabel_dir, word_to_id, cat_to_id, wordslength)

        for i in range(np.shape(allunlabel)[0]):
            unlabel_name.append(allunlabel[i][0])
            # for j in range(np.shape(allunlabel)[1]-1):
            #     unlabel_data[i][j] = allunlabel[i][j+1].split(':')[1]

        x = np.vstack(x_train, x_unlabel)
        y = np.hstack(y_train, [None]*len(y_unlabel))

        trn_ds = Dataset(x, y)
        real_trn_ds = Dataset(x_train, y_train)

        #返回训练集，训练集的个数，未标记数据的名称list
        return trn_ds, numoftrain, unlabel_name, real_trn_ds

    def split_train_test_rnn(self, train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, n_labeled, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)

        # train, test, unlabel通用
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)

        # 处理train
        data_id_train, label_id_train = process_file_rnn(train_dir, word_to_id, cat_to_id, 0,  wordslength)
        data_id_test, label_id_test = process_file_rnn(test_dir, word_to_id, cat_to_id, 0,  wordslength)
        data_id_unlabel, label_id_unlabel, unlabelnames = process_file_rnn(test_dir, word_to_id, cat_to_id, 1,  wordslength)

        # 处理train
        y_train_temp = kr.utils.to_categorical(label_id_train, num_classes=len(cat_to_id))

        X_train = kr.preprocessing.sequence.pad_sequences(data_id_train, wordslength)
        Y_train = self.convertlabel(y_train_temp)
        # 处理test
        y_test_temp = kr.utils.to_categorical(label_id_test, num_classes=len(cat_to_id))

        X_test_temp = kr.preprocessing.sequence.pad_sequences(data_id_test, wordslength)
        Y_test_temp = self.convertlabel(y_test_temp)

        # 处理 Unlabel
        X_unlabel = kr.preprocessing.sequence.pad_sequences(data_id_unlabel, wordslength)
        Y_unlabel = np.array(label_id_unlabel)

        # 将test集拆分成test 和val
        X_test, X_val, Y_test, Y_val = train_test_split(X_test_temp, Y_test_temp, test_size=0.8)

        # [TODO]将unlabel集label打为NONE同时与Train拼接
        trn_ds = Dataset(np.concatenate(X_train, X_unlabel), np.concatenate(Y_train, [None] * len (Y_unlabel)))
        val_ds = Dataset(X_val, Y_val)
        tst_ds = Dataset(X_test, Y_test)

        draw_ds = Dataset(np.concatenate(X_train, X_test, X_val), np.concatenate(Y_train, [NONE] * (len(Y_test) + len(Y_val))))
        return trn_ds, val_ds, tst_ds, draw_ds, unlabelnames

        # [TODO]将普通AL 从DAL中拆分出来并生成特定的函数 > DONE

        # [TODO]写Label的检测函数，确认Label的数量

        # [TODO]返回包括Unlabel Names

    def split_train_test_tal(self, train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, n_labeled, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)
            categories, cat_to_id = read_category()
            words, word_to_id = read_vocab(vocab_dir)

            data_id_train, label_id_train = process_file(train_dir, word_to_id, cat_to_id, 0, wordslength)
            data_id_test, label_id_test = process_file(test_dir, word_to_id, cat_to_id, 0, wordslength)
            data_id_unlabel, label_id_unlabel, unlabelnames = process_file(test_dir, word_to_id, cat_to_id, 1, wordslength)
            # x, y = process_file(train_dir, word_to_id, cat_to_id, wordslength)
            # x_rnn, y_rnn = process_file_rnn(train_dir, word_to_id, cat_to_id, 600)

            X_train = data_id_train
            Y_train = self.convertlabel(label_id_train)

            X_test = data_id_test
            Y_test = self.convertlabel(label_id_test)

            X_unlabel = data_id_unlabel
            Y_unlabel = np.array(label_id_unlabel)

            trn_ds = Dataset(np.concatenate(X_train, X_unlabel), np.concatenate(Y_train, [None] * len(Y_unlabel)))
            tst_ds = Dataset(X_test, Y_test)

            #[todo]将X_train和X_test组合，随机划分部分测试集，做空跑画图像显示效果用

            draw_ds = Dataset(np.concatenate(X_train, X_test, X_val),
                              np.concatenate(Y_train, [NONE] * (len(Y_test) + len(Y_val))))
            return trn_ds, val_ds, tst_ds, draw_ds, unlabelnames

    def split_onlytest(self, test_dir, vocab_dir, wordslength):
        #返回测试集样例
        # dataset_train = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), localdir)
        # dataset_train = zipfile.ZipFile(docfile).read(localdir).decode('utf-8')

        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab(vocab_dir)
        x, y = process_file(test_dir, word_to_id, cat_to_id, wordslength)

        listy = []
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if y[i][j] == 1:
                    listy.append(j)
        listy = np.array(listy)
        X_test, X_train, y_test, y_train = \
            train_test_split(x, listy, test_size=1)

        test_ds = Dataset(X_test, y_test)
        return test_ds
        # x, y = import_libsvm_sparse(dataset_train).format_sklearn()
        # test_ds = Dataset(x,y)
        # return test_ds

    # 【上传只有train】，随机划分一部分做test
    def split_train_test (self, train_dir, vocab_dir, test_size, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, 1000)

        categories, cat_to_id = read_category()
        words, word_to_id = read_vocab(vocab_dir)

        x, y = process_file(train_dir, word_to_id, cat_to_id, wordslength)
        # x_rnn, y_rnn = process_file_rnn(train_dir, word_to_id, cat_to_id, 600)

        listy = []
        for i in range(np.shape(y)[0]):
            for j in range(np.shape(y)[1]):
                if y[i][j] == 1:
                    listy.append(j)
        listy = np.array(listy)

        # dataset_train = train_dir
        # x, y = import_libsvm_sparse(dataset_train).format_sklearn()

        X_train, X_test, y_train, y_test = train_test_split(x, listy, test_size=test_size)

        #train_ds = Dataset(X_train,label_train)
        fully_tst_ds = Dataset(X_test, y_test)

        X_val, X_real_test, y_val, y_real_test = \
            train_test_split(X_test, y_test, test_size=0.5)

        tst_ds = Dataset(X_real_test, y_real_test)
        val_ds = Dataset(X_val, y_val)

        # fully_labeled_trn_ds = Dataset(X_train, y_train)

        return X_train, y_train, fully_tst_ds, tst_ds, val_ds

    # 把Train.txt和Unlabel.txt 两个文件结合起来，形成一个
    def split_train_test_unlabel(self, train, trainlabel, unlabeltext):

        #原训练集
        numoftrain = len(train)
        #未标记集
        allunlabel = np.loadtxt(unlabeltext, dtype = str)

        #减1减去的是unlabel向量的[名称]
        unlabel_data = np.zeros((np.shape(allunlabel)[0], np.shape(allunlabel)[1]-1))
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

    def sendfile(self, filedir, filetype, username, useremail, password, numneedtobemarked):
        SIZE = 65535
        RECSIZE = 1024
        jsontosend = {}
        id = time.strftime("%Y", time.localtime())+time.strftime("%m", time.localtime())+time.strftime("%d", time.localtime())+time.strftime("%H", time.localtime())+time.strftime("%M", time.localtime())+time.strftime("%S", time.localtime())
        id = id + str(random.randint(0, 9))+str(random.randint(0, 9))+str(random.randint(0, 9))+str(random.randint(0, 9))+str(random.randint(0, 9))
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

    # get the min nums of labeled initially
    def getlabelnum(self, trn_ds):
        notolabel = [] # 记录需要标注的序号，将其余的元素置为None
        initlabel = []
        initcontents = []
        maxinitnum = 2 # 每个class允许的最大初始标记个数

        contents, labels = read_file(trn_ds)
        distinctlabel = list(set(labels))
        for i in distinctlabel:
            for j in labels:
                if j == i:
                    contents.pop(labels.index(j))
                    labels.pop(j)
                    initcontents.append(labels.index(j))
                    initlabel.append(j)
                    maxinitnum = maxinitnum - 1
                    if maxinitnum == 0:
                        maxinitnum = 2
                        break
                    else:
                        pass
        initdataset = Dataset(np.concatenate(initcontents, contents), np.concatenate(initlabel, len(initcontents)*[None]))

        return initdataset, len(initlabel), len(labels)

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
                    flag = 1
                    n_labeled = i + 2
                    break
                flag = -1
        if flag == -1:
            pass#[todo]报错


        none_trn_ds = Dataset(x_train, np.concatenate(
            [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

        return none_trn_ds


    #空跑测试集做图像
    def score_ideal(self, trn_ds, tst_ds, lbr, model, qs, quota):
        E_in, E_out = [], []

        for _ in range(quota):
            ask_id = qs.make_query()

            X, _ = zip(*trn_ds.data)
            c = len(X)
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)
            model.train(trn_ds)
            E_in = np.append(E_in, 1 - model.score(trn_ds)) #in-sample error
            E_out = np.append(E_out, 1 - model.score(tst_ds)) #out-sample error

        return E_in, E_out

    def plotforimage(self, query_num, E_in1, E_in2, E_out1, E_out2, username):
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

    def myRegression(self, algorithm, trn_ds, none_trn_ds):
        if algorithm == 'qbc':
            qs = QueryByCommitteePlus(trn_ds, models=[LogisticRegression(C=1.0), LogisticRegression(C=0.4), ], )
            qs_fordraw = QueryByCommittee(none_trn_ds,
                                          models=[LogisticRegression(C=1.0), LogisticRegression(C=0.4), ], )
        elif algorithm == 'us':
            qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
            qs_fordraw = UncertaintySampling(none_trn_ds, method='lc', model=LogisticRegression())
        elif algorithm == 'albc':
            qs = ActiveLearningByLearningPlus(trn_ds, query_strategies=[
                UncertaintySampling(trn_ds, method='lc', model=LogisticRegression()),
                QueryByCommittee(trn_ds, models=[LogisticRegression(C=1.0), LogisticRegression(C=0.4), ], ), ],
                                              T=quota,
                                              uniform_sampler=True,
                                              model=LogisticRegression()
                                              )
            qs_fordraw = ActiveLearningByLearning(none_trn_ds,
                                                  query_strategies=[UncertaintySampling(none_trn_ds, method='lc',
                                                                                        model=LogisticRegression()),
                                                                    QueryByCommittee(none_trn_ds,
                                                                                     models=[LogisticRegression(C=1.0),
                                                                                             LogisticRegression(
                                                                                                 C=0.4), ], ), ],
                                                  T=quota,
                                                  uniform_sampler=True,
                                                  model=LogisticRegression()
                                                  )
        elif algorithm == 'dal':
            pass
            #  [todo] add dal
        else:
            pass

        return qs, qs_fordraw

    def svmClassify(self, algorithm, trn_ds, none_trn_ds):
        if algorithm == 'qbc':
            qs = QueryByCommitteePlus(trn_ds, models=[SVM(C=1.0, decision_function_shape='ovr'),
                                                      SVM(C=0.4, decision_function_shape='ovr'), ], )
            qs_fordraw = QueryByCommittee(none_trn_ds, models=[SVM(C=1.0, decision_function_shape='ovr'),
                                                               SVM(C=0.4, decision_function_shape='ovr'), ], )
        elif algorithm == 'us':
            qs = UncertaintySampling(trn_ds, model=SVM(decision_function_shape='ovr'))
            qs_fordraw = UncertaintySampling(none_trn_ds, model=SVM(decision_function_shape='ovr'))
        elif algorithm == 'albl':
            qs = ActiveLearningByLearningPlus(trn_ds, query_strategies=[
                UncertaintySampling(trn_ds, model=SVM(decision_function_shape='ovr')), QueryByCommittee(trn_ds, models=[
                    SVM(C=1.0, decision_function_shape='ovr'), SVM(C=0.4, decision_function_shape='ovr'), ], ), ],
                                              T=quota,
                                              uniform_sampler=True,
                                              model=SVM(kernel='linear', decision_function_shape='ovr')
                                              )
            qs_fordraw = ActiveLearningByLearning(none_trn_ds, query_strategies=[
                UncertaintySampling(none_trn_ds, model=SVM(decision_function_shape='ovr')),
                QueryByCommittee(none_trn_ds, models=[SVM(C=1.0, decision_function_shape='ovr'),
                                                      SVM(C=0.4, decision_function_shape='ovr'), ], ), ],
                                                  T=quota,
                                                  uniform_sampler=True,
                                                  model=SVM(kernel='linear', decision_function_shape='ovr')
                                                  )
        elif algorithm == 'dal':
            pass
            # [todo] add dal
        else:
            pass

        return qs, qs_fordraw

    def maintodo(self, kind, modelselect, strategy, algorithm, quota, trainAndtest, testsize, pushallask,docfile,username,useremail,bmtpassword):
        zipfile.ZipFile(docfile).extractall('/app/codalab/thirdpart/'+username)
        trainentity = '/app/codalab/thirdpart/' + username + '/train.txt'
        unlabelentity = '/app/codalab/thirdpart/' + username + '/unlabel.txt'
        testentity = '/app/codalab/thirdpart/' + username + '/test.txt'
        vocab_dir = '/app/codalab/thirdpart/' + username + '/vocab/vocab.txt'

        # 提交未标记集的两种模式，一种有unlabel文件夹，一种有unlabel文件
        unlabeldatasetdir = '/app/codalab/thirdpart/' + username + '/unlabel/'
        vocab_size = 1000

        # [Todo]:需要标记的个数,是否提交的是训练集+测试集，如果只提交一个测试集，那么划分训练集的比例，
        #1是提交的两个文件，一个训练集一个测试集。
        # trainAndtest = 1
        # testsize = 0.33
        # 1是一次提供所有askid, 0是交互式问询
        # pushallask = 1
        #如果是交互式问询的话，那么需要标记的标记列表
        #interlabel = ['0','1']
        askidlist = []

        # Todo:ask_id是问的train+unlabel中unlabel的id



        # unlabeldatasetdir = os.listdir(unlabeldatasetdir)

        E_in1, E_in2 = [], []
        E_out1, E_out2 = [], []
        unlabeldict = {}

        #提交的是Train还是Train+Test
        if trainAndtest == 1:
            trn_ds, val_ds, tst_ds, draw_ds, unlabelnames = self.split_train_test_tal(train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, n_labeled, wordslength)
            trn_ds_for_draw, initlabelednumbers, quota = self.getlabelnum(draw_ds)
        # 暂时取消提交只有一个Train的逻辑，强行规定必须划分Test哪怕随机划分也好
        else:
            pass
            # 提交的只有一个Train
            # X_train, y_train, fully_tst_ds, tst_ds, val_ds = self.split_train_test(trainentity, testsize)
            # trn_ds, numoftrain, unlabelnames = self.split_train_test_unlabel(X_train, y_train, unlabelentity)
            # real_trn_ds = copy.deepcopy(trn_ds)
            # none_trn_ds = self.split_for_drawplot(real_trn_ds, numoftrain, quota)


        trn_ds_random = copy.deepcopy(trn_ds_for_draw) # 原验证集强行划分部分未知None做效果用
        qs_random = RandomSampling(trn_ds_random)

# ========================Binary====================
        if strategy == 'binary':
            if modelselect == 'logic':
                qs, qs_fordraw = self.myRegression(algorithm, trn_ds, none_trn_ds)
                model = LogisticRegression()
                lbr = IdealLabeler(real_trn_ds)
                E_in1, E_out1 = self.score_ideal(none_trn_ds, tst_ds, lbr, model, qs_fordraw, quota)
                model = LogisticRegression()
                E_in2, E_out2 = self.score_ideal(trn_ds_random, tst_ds, lbr, model, qs_random, quota)
            elif modelselect == 'svm':
                qs, qs_fordraw = self.svmClassfiy(algorithm, trn_ds, none_trn_ds)
                lbr = IdealLabeler(real_trn_ds)
                model = SVM(kernel='linear', decision_function_shape='ovr')
                E_in1, E_out1 = self.score_ideal(none_trn_ds, tst_ds, lbr, model, qs_fordraw, quota)
                model = SVM(kernel='linear', decision_function_shape='ovr')
                E_in2, E_out2 = self.score_ideal(trn_ds_random, tst_ds, lbr, model, qs_random, quota)
            else:
                pass

        elif strategy == 'multiclass':
            if modelselect == 'svm':
                qs, qs_fordraw = self.svmClassfiy(algorithm, trn_ds, none_trn_ds)
                lbr = IdealLabeler(real_trn_ds)
                model = SVM(kernel='linear', decision_function_shape='ovr')
                E_in1, E_out1 = self.score_ideal(none_trn_ds, tst_ds, lbr, model, qs_fordraw, quota)
                model = SVM(kernel='linear', decision_function_shape='ovr')
                E_in2, E_out2 = self.score_ideal(trn_ds_random, tst_ds, lbr, model, qs_random, quota)
            else:
                pass

        elif strategy == 'multilabel':
            pass

        else:
            pass



        self.plotforimage(np.arange(1, quota + 1), E_in1, E_in2, E_out1, E_out2, username)


#dir = '/app/codalab/static/img/partpicture/'+username+'/'
        #返回一批实例,返回分数是为了解决不标注的情况下无法自动更新的问题
        if pushallask == 1:

            first, scores = qs.make_query(return_score = True)
            number, num_score = zip(*scores)[0], zip(*scores)[1]
            num_score_array = np.array(num_score)
            max_n = heapq.nlargest(quota, range(len(num_score_array)), num_score_array.take)

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

                return askidlist, csvdir
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
                    writer.writerow(['name', 'entity'])
                    for key, value in unlabeldict.items():
                        #askidlist.append(key)
                        writer.writerow([key, value])
                return askidlist,csvdir


        #向标注平台发送
        else:
            first, scores = qs.make_query(return_score = True)
            number, num_score = zip(*scores)[0], zip(*scores)[1]
            num_score_array = np.array(num_score)
            max_n = heapq.nlargest(quota, range(len(num_score_array)),num_score_array.take)
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
