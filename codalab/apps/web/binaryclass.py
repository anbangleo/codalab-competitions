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
plt.switch_backend('Agg')
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

from dealwordindict import read_vocab, read_category, batch_iter, process_file, process_file_rnn, build_vocab, native_content, read_file, read_file_nocut
import time
from datetime import timedelta
import heapq
from rnnmodel import RNN_Probability_Model
from rnn_model_config import TRNNConfig
from cnnmodel import CNN_Probability_Model
import random
import codecs

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
        data_id, label_id = process_file_rnn(train_dir, word_to_id, cat_to_id, wordslength)
        data_id_test, label_id_test = process_file_rnn(test_dir, word_to_id, cat_to_id, 0, wordslength)
        data_id_unlabel, label_id_unlabel, unlabelcontents, unlabelnames = process_file_rnn(unlabel_dir, word_to_id,
                                                                                            cat_to_id, 1, wordslength)


    def split_train_test_dal(self,train_dir, test_dir, vocab_dir, vocab_size, n_labeled, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)

        # train, test, unlabel通用
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)

        data_id_train, label_id_train = process_file_rnn(train_dir, word_to_id, cat_to_id, wordslength)
        data_id_test, label_id_test = process_file_rnn(test_dir, word_to_id, cat_to_id, wordslength)
        # x_rnn, y_rnn = process_file_rnn(train_dir, word_to_id, cat_to_id, 600)
        data_id_train.extend(data_id_test)
        label_id_train.extend(label_id_test)
        y = kr.utils.to_categorical(label_id_train, num_classes=len(cat_to_id))
        listy = self.convertlabel(y)

        X_train, X_test, y_train, y_test = \
            train_test_split(data_id_train, listy, test_size=0.2)

        X_train_al = []
        X_test_al = []
        res = []
        for i in X_train:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_train_al.append(res)
            res = []

        for i in X_test:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_test_al.append(res)
            res = []

        X_train_al = np.array(X_train_al)
        X_test_al = np.array(X_test_al)


        trn_ds_al = Dataset(X_train_al, np.concatenate(
            [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))

        tst_ds_al = Dataset(X_test_al, y_test)

        X_train_rnn = kr.preprocessing.sequence.pad_sequences(X_train, wordslength)
        X_test_rnn = kr.preprocessing.sequence.pad_sequences(X_test, wordslength)

        X_train_rnn, X_val_rnn, y_train_rnn, y_val_rnn = \
            train_test_split(X_train_rnn, y_train, test_size=0.2)

        trn_ds_rnn = Dataset(X_train_rnn, np.concatenate(
            [y_train_rnn[:n_labeled], [None] * (len(y_train_rnn) - n_labeled)]))

        val_ds_rnn = Dataset(X_val_rnn, y_val_rnn)

        tst_ds_rnn = Dataset(X_test_rnn, y_test)

        fully_labeled_trn_ds_al = Dataset(X_train_al, y_train)
        fully_labeled_trn_ds_rnn = Dataset(X_train_rnn, y_train_rnn)

        return trn_ds_al, tst_ds_al, y_train_rnn, fully_labeled_trn_ds_al, \
               trn_ds_rnn, tst_ds_rnn, fully_labeled_trn_ds_rnn, val_ds_rnn

    def split_train_test_rnn(self, train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)

        # train, test, unlabel通用
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)

        # 处理train
        data_id_train, label_id_train = process_file_rnn(train_dir, word_to_id, cat_to_id, 0,  wordslength)
        data_id_test, label_id_test = process_file_rnn(test_dir, word_to_id, cat_to_id, 0,  wordslength)
        data_id_unlabel, label_id_unlabel, unlabelcontents, unlabelnames = process_file_rnn(unlabel_dir, word_to_id, cat_to_id, 1,  wordslength)

        # 处理train
        y_train_temp = kr.utils.to_categorical(label_id_train, num_classes=len(cat_to_id))

        X_train = kr.preprocessing.sequence.pad_sequences(data_id_train, wordslength)
        Y_train = self.convertlabel(y_train_temp)
        # 处理test
        y_test_temp = kr.utils.to_categorical(label_id_test, num_classes=len(cat_to_id))

        X_test = kr.preprocessing.sequence.pad_sequences(data_id_test, wordslength)
        Y_test = self.convertlabel(y_test_temp)

        # 处理 Unlabel
        X_unlabel = kr.preprocessing.sequence.pad_sequences(data_id_unlabel, wordslength)
        Y_unlabel = np.array(label_id_unlabel)

        # 将test集拆分成test 和val
        X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.2)

        trn_ds = Dataset(X_train, Y_train)
        tst_ds = Dataset(X_test, Y_test)
        val_ds = Dataset(X_val, Y_val)
        # ===================以上是处理正式数据集的================
        # ===================以下是划分效果集供跑效果用的================
        data_id_train_fordraw, data_id_test_fordraw, label_id_train_fordraw, label_id_test_fordraw = train_test_split(
            np.concatenate([data_id_train, data_id_test]), np.concatenate([label_id_train, label_id_test]), test_size=0.3)
        data_id_test_fordraw, data_val_fordraw, label_id_test_fordraw, label_val_fordraw = train_test_split(
            data_id_test_fordraw, label_id_test_fordraw,  test_size=0.3)

        # X_train_fordraw, X_test_fordraw, Y_train_fordraw, Y_test_fordraw = train_test_split(
        #     np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), test_size=0.3)
        # X_test_fordraw, X_val_fordraw, Y_test_fordraw, Y_val_fordraw = train_test_split(
        #     X_test_fordraw, Y_test_fordraw,  test_size=0.3)
        # -------------------处理DAL的部分-------------------
        X_train_fordraw_dal = kr.preprocessing.sequence.pad_sequences(data_id_train_fordraw, wordslength)
        y_train_temp = kr.utils.to_categorical(label_id_train_fordraw, num_classes=len(cat_to_id))
        Y_train_fordraw_dal = self.convertlabel(y_train_temp)
        X_test_fordraw_dal = kr.preprocessing.sequence.pad_sequences(data_id_test_fordraw, wordslength)
        y_train_temp = kr.utils.to_categorical(label_id_test_fordraw, num_classes=len(cat_to_id))
        Y_test_fordraw_dal = self.convertlabel(y_train_temp)
        X_val_fordraw_dal = kr.preprocessing.sequence.pad_sequences(data_val_fordraw, wordslength)
        y_train_temp = kr.utils.to_categorical(label_val_fordraw, num_classes=len(cat_to_id))
        Y_val_fordraw_dal = self.convertlabel(y_train_temp)
        tst_fordraw_dal = Dataset(X_test_fordraw_dal, Y_test_fordraw_dal)
        val_fordraw = Dataset(X_val_fordraw_dal, Y_val_fordraw_dal)
        # -------------------处理TAL的部分-------------------
        X_train_fordraw_tal = []
        X_test_fordraw_tal = []
        res = []
        for i in data_id_train_fordraw:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_train_fordraw_tal.append(res)
            res = []
        res = []
        for i in data_id_test_fordraw:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_test_fordraw_tal.append(res)
            res = []

        # X_train = data_id_train
        X_train_fordraw_tal = np.array(X_train_fordraw_tal)
        Y_train_fordraw_tal = np.array(label_id_train_fordraw)

        # X_test = data_id_test
        X_test_fordraw_tal = np.array(X_test_fordraw_tal)
        Y_test_fordraw_tal = np.array(label_id_test_fordraw)
        tst_fordraw_tal = Dataset(X_test_fordraw_tal,Y_test_fordraw_tal)


        # 开始处理X_train_fordraw为了画图
        distinctlabel_train = list(set(Y_train_fordraw_tal))
        distinctlabel_test = list(set(Y_test_fordraw_tal))
        numofclasses_train = len(distinctlabel_train)
        numofclasses_test = len(distinctlabel_test)
        initcontents_tal = []
        initlabel_tal = []
        initcontents_dal = []
        initlabel_dal = []
        # initlabel = np.array(initlabel)
        # initcontents = np.array(initcontents)
        # [TODO] check the nums equal
        maxinitnum = (len(X_train_fordraw_tal)/numofclasses_train)/3
        maxinitnums = maxinitnum
        for i in distinctlabel_train:
            for j in range(len(Y_train_fordraw_tal)):
                if Y_train_fordraw_tal[j] == i:
                    initcontents_tal.append(X_train_fordraw_tal[j])
                    initlabel_tal.append(Y_train_fordraw_tal[j])
                    initcontents_dal.append(X_train_fordraw_dal[j])
                    initlabel_dal.append(Y_train_fordraw_dal[j])
                    X_train_fordraw_dal = np.delete(X_train_fordraw_dal, j, axis=0)
                    Y_train_fordraw_dal = np.delete(Y_train_fordraw_dal, j, axis=0)
                    X_train_fordraw_tal = np.delete(X_train_fordraw_tal, j, axis=0)
                    Y_train_fordraw_tal = np.delete(Y_train_fordraw_tal, j, axis=0)
                    maxinitnums = maxinitnums - 1
                    if maxinitnums == 0:
                        maxinitnums = maxinitnum
                        break
                    else:
                        pass
        initlabel_tal = np.array(initlabel_tal)
        initcontents_tal = np.array(initcontents_tal)
        initlabel_dal = np.array(initlabel_dal)
        initcontents_dal = np.array(initcontents_dal)
        # trn_ds_fordraw_fully = Dataset(X_train_fordraw, Y_train_fordraw)

        trn_ds_fordraw_fully_tal = Dataset(np.concatenate([initcontents_tal, X_train_fordraw_tal]),
                                       np.concatenate([initlabel_tal, Y_train_fordraw_tal]))
        trn_ds_fordraw_none_tal = Dataset(np.concatenate([initcontents_tal, X_train_fordraw_tal]),
                                      np.concatenate([initlabel_tal, [None] * len(Y_train_fordraw_tal)]))

        trn_ds_fordraw_fully_dal = Dataset(np.concatenate([initcontents_dal, X_train_fordraw_dal]),
                                           np.concatenate([initlabel_dal, Y_train_fordraw_dal]))
        trn_ds_fordraw_none_dal = Dataset(np.concatenate([initcontents_dal, X_train_fordraw_dal]),
                                          np.concatenate([initlabel_dal, [None] * len(Y_train_fordraw_dal)]))


        quota_fordraw = len(Y_train_fordraw_tal)

        # draw_ds = Dataset(np.concatenate(X_train, X_test, X_val),
        #                   np.concatenate(Y_train, [NONE] * (len(Y_test) + len(Y_val))))
        return trn_ds, tst_ds, val_ds, unlabelcontents, unlabelnames, \
        trn_ds_fordraw_fully_dal, trn_ds_fordraw_none_dal, tst_fordraw_dal, val_fordraw,\
        trn_ds_fordraw_fully_tal, trn_ds_fordraw_none_tal, tst_fordraw_tal, \
        quota_fordraw, categories, len(categories)


        # [TODO]将普通AL 从DAL中拆分出来并生成特定的函数 > DONE

        # [TODO]写Label的检测函数，确认Label的数量

        # [TODO]返回包括Unlabel Names

    def split_train_test_tal(self, train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, maxinitnum, wordslength):
        if not os.path.exists(vocab_dir):
            build_vocab(train_dir, vocab_dir, vocab_size, unlabel_dir, test_dir)
        categories, cat_to_id = read_category(train_dir)
        words, word_to_id = read_vocab(vocab_dir)

        data_id_train, label_id_train = process_file_rnn(train_dir, word_to_id, cat_to_id, 0, wordslength)
        data_id_test, label_id_test = process_file_rnn(test_dir, word_to_id, cat_to_id, 0, wordslength)
        data_id_unlabel, label_id_unlabel, unlabelcontents, unlabelnames = process_file_rnn(unlabel_dir, word_to_id, cat_to_id, 1, wordslength)
        # x, y = process_file(train_dir, word_to_id, cat_to_id, wordslength)
        # x_rnn, y_rnn = process_file_rnntrain_dir(train_dir, word_to_id, cat_to_id, 600)
        X_train = []
        X_test = []
        X_unlabel = []
        res = []
        maxinitnums = maxinitnum

        for i in data_id_train:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_train.append(res)
            res = []
        res = []
        for i in data_id_test:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_test.append(res)
            res = []
        res = []
        for i in data_id_unlabel:
            for j in range(wordslength):
                a = i.count(j)
                if a > 0:
                    res.append(a)
                else:
                    res.append(0)
            X_unlabel.append(res)
            res = []


        # X_train = data_id_train
        Y_train = np.array(label_id_train)

        # X_test = data_id_test
        Y_test = np.array(label_id_test)

        # X_unlabel = data_id_unlabel
        Y_unlabel = np.array(label_id_unlabel)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_unlabel = np.array(X_unlabel)

        trn_ds = Dataset(np.concatenate([X_train, X_unlabel]), np.concatenate([Y_train, [None] * len(Y_unlabel)]))
        tst_ds = Dataset(X_test, Y_test)

        X_train_fordraw, X_test_fordraw, Y_train_fordraw, Y_test_fordraw = train_test_split(np.concatenate([X_train, X_test]), np.concatenate([Y_train, Y_test]), test_size = 0.3)


        # X_train_fordraw_copy = copy.deepcopy(X_train_fordraw)
        # Y_train_fordraw_copy = copy.deepcopy(Y_train_fordraw)
        # 开始处理X_train_fordraw为了画图
        distinctlabel_train = list(set(Y_train_fordraw))
        distinctlabel_test = list(set(Y_test_fordraw))
        numofclasses_train = len(distinctlabel_train)
        numofclasses_test = len(distinctlabel_test)
        initcontents = []
        initlabel = []
        # initlabel = np.array(initlabel)
        # initcontents = np.array(initcontents)
        # [TODO] check the nums equal

        for i in distinctlabel_train:
            for j in range(len(Y_train_fordraw)):
                if Y_train_fordraw[j] == i:
                    initcontents.append(X_train_fordraw[j])
                    initlabel.append(Y_train_fordraw[j])

                    # initcontents = np.append(initcontents, X_train_fordraw[j])
                    # initlabel = np.append(initlabel, Y_train_fordraw[j])
                    X_train_fordraw = np.delete(X_train_fordraw, j, axis=0)
                    Y_train_fordraw = np.delete(Y_train_fordraw, j, axis=0)
                    maxinitnums = maxinitnums - 1
                    if maxinitnums == 0:
                        maxinitnums = maxinitnum
                        break
                    else:
                        pass
        initlabel = np.array(initlabel)
        initcontents = np.array(initcontents)

        # trn_ds_fordraw_fully = Dataset(X_train_fordraw, Y_train_fordraw)

        trn_ds_fordraw_fully = Dataset(np.concatenate([initcontents, X_train_fordraw]), np.concatenate([initlabel, Y_train_fordraw]))
        trn_ds_fordraw_none = Dataset(np.concatenate([initcontents, X_train_fordraw]), np.concatenate([initlabel, [None] * len(Y_train_fordraw)]))

        tst_ds_fordraw = Dataset(X_test_fordraw, Y_test_fordraw)
        quota_fordraw = len(Y_train_fordraw)

            # draw_ds = Dataset(np.concatenate(X_train, X_test, X_val),
            #                   np.concatenate(Y_train, [NONE] * (len(Y_test) + len(Y_val))))
        return trn_ds, tst_ds, unlabelcontents, unlabelnames, trn_ds_fordraw_fully, trn_ds_fordraw_none, tst_ds_fordraw, quota_fordraw

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


    # get the min nums of labeled initially, *args是可能的tst_dir等
    def getlabelnum(self, trn_dir, *args):
        notolabel = [] # 记录需要标注的序号，将其余的元素置为None
        initlabel = []
        initcontents = []
        maxinitnum = 2 # 每个class允许的最大初始标记个数

        contents, labels = read_file(trn_dir)
        distinctlabel = list(set(labels)) # multi-class的多类标签
        distinctlabel.sort()
        for i in distinctlabel:
            for j in range(len(labels)):
                if labels[j] == i:
                    contents.pop(j)
                    labels.pop(j)
                    initcontents.append(contents[j])
                    initlabel.append(labels[j])
                    maxinitnum = maxinitnum - 1
                    if maxinitnum == 0:
                        maxinitnum = 2
                        break
                    else:
                        pass
        initdataset = Dataset(np.concatenate([initcontents, contents]), np.concatenate([initlabel, len(labels)*[None]]))
        n_labeled_initial = len(initlabel)
        quota_initial = len(labels)
        # 如果传入了tst_dir
        if len(args) == 1:
            tst_dir = args[0]
            contents_tst, labels_tst = read_file(tst_dir)
            distinctlabel_tst = list(set(labels_tst))
            distinctlabel_tst.sort()
            if len(distinctlabel) != len(distinctlabel_tst):
                ErrorMessage = 'Oppos! The Class Names/Numbers is different between train_dataset and test_dataset.'
                return ErrorMessage
            else:
                for i in distinctlabel:
                    if i in distinctlabel_tst:
                        pass
                    else:
                        ErrorMessage = 'Oppos! The Class Names/Numbers is different between train_dataset and test_dataset.'
                        return ErrorMessage
            initdataset = Dataset(np.concatenate([initcontents, contents, contents_tst]), np.concatenate([initlabel, [None]*(len(labels) + len(labels_tst))]))
            quota_initial = len(labels) + len(labels_tst)


        # 对于模拟的图像，initlabel是初始标记的个数，labels一定是all-initlabel(初始数据集要求全ideal标记)
        return initdataset, len(initlabel), len(labels)

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

    def realrun_random(self, trn_ds, tst_ds, lbr, model, qs, quota, batchsize):
        E_in, E_out = [], []
        intern = 0
        finalnum = 0
        # print ("[Important] Start the Random Train:")
        # start_time = time.time()
        if quota % batchsize == 0:
            intern = int(quota / batchsize)
        else:
            intern = int(quota / batchsize) + 1
            finalnum = int(quota % batchsize)

        for t in range(intern):
            unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
            if t == intern - 1 and finalnum != 0:
                max_n = random.sample(unlabeled_entry_ids, finalnum)
            else:
                max_n = random.sample(unlabeled_entry_ids, batchsize)

            X, _ = zip(*trn_ds.data)
            for ask_id in max_n:
                lb = lbr.label(X[ask_id])
                trn_ds.update(ask_id, lb)

            model.train(trn_ds)
            E_in = np.append(E_in, 1 - model.score(trn_ds))
            E_out = np.append(E_out, 1 - model.score(tst_ds))
            # print (E_out)

        # E_time = get_time_dif(start_time)
        # print("time to train" + str(time_dif))

        return E_in, E_out

    def realrun_qs(self, trn_ds, tst_ds, lbr, model, qs, quota, batchsize):
        E_in, E_out = [], []
        intern = 0
        finalnum = 0

        # [TODO] fix the issue
        # test_trnds = trn_ds.get_labeled_entries()
        # print ttt

        # print ("[Important] Start the UncertaintySampling Train:")
        # start_time = time.time()
        if quota % batchsize == 0:
            intern = int(quota / batchsize)
        else:
            intern = int(quota / batchsize) + 1
            finalnum = int(quota % batchsize)

        for t in range(intern):
            # print ("[QS]this is the " + str(t) + " times to ask")
            first, scores = qs.make_query(return_score=True)

            # python 2&&python3
            number, num_score = zip(*scores)[0], zip(*scores)[1]
            # num_score = next(zip*scores)
            # itscore = zip(*scores)
            # number = next(itscore)
            # num_score = next(itscore)

            num_score_array = np.array(num_score)
            # max_n = heapq.nlargest(quota, range(len(num_score_array)), num_score_array.take)

            if t == intern - 1 and finalnum != 0:
                max_n = heapq.nlargest(finalnum, range(len(num_score_array)), num_score_array.take)
            else:
                max_n = heapq.nlargest(batchsize, range(len(num_score_array)), num_score_array.take)

            unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())

            X, _ = zip(*trn_ds.data)
            # print (max_n)
            for ask_id in max_n:
                real_id = unlabeled_entry_ids[ask_id]
                lb = lbr.label(X[real_id])
                trn_ds.update(real_id, lb)

            model.train(trn_ds)

            E_in = np.append(E_in, 1 - model.score(trn_ds))
            E_out = np.append(E_out, 1 - model.score(tst_ds))
        # E_time = get_time_dif(start_time)

        return E_in, E_out

    def realrun_dal(self, trn_ds, tst_ds, val_ds, lbr, model, quota, best_val, batchsize):
        E_in, E_out = [], []
        intern = 0
        finalnum = 0
        # start_time = time.time()
        if quota % batchsize == 0:
            intern = int(quota / batchsize)
        else:
            intern = int(quota / batchsize) + 1
            finalnum = int(quota % batchsize)
        # 跑真实的数据，只要返回TOP N 个未标记集即可
        if lbr == 'realrun':
            scores = model.predict_pro(trn_ds)
            unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
            max_n = heapq.nsmallest(quota, range(len(scores)), scores.take)
            return max_n
        # 跑效果数据，真实的情况
        else:
            for t in range(intern):
                x_first_train = []
                y_first_train = []

                scores = model.predict_pro(trn_ds)

                unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())

                if t == intern - 1 and finalnum != 0:
                    max_n = heapq.nsmallest(finalnum, range(len(scores)), scores.take)
                else:
                    max_n = heapq.nsmallest(batchsize, range(len(scores)), scores.take)

                X, _ = zip(*trn_ds.data)

                for ask_id in max_n:
                    real_id = unlabeled_entry_ids[ask_id]
                    lb = lbr.label(X[real_id])
                    trn_ds.update(real_id, lb)
                    x_first_train.append(X[real_id])
                    y_first_train.append(lb)

                x_first_train = np.array(x_first_train)
                y_first_train = np.array(y_first_train)

                first_train = Dataset(x_first_train, y_first_train)

                best_val = model.retrain(trn_ds, val_ds, best_val, first_train)

                # E_in = np.append(E_in, 1 - model.score(trn_ds))
                E_out = np.append(E_out, 1 - model.score(tst_ds))

            # E_time = get_time_dif(start_time)
            return E_out

    def plotforimage(self, query_num, E_in1, E_in2, E_out1, E_out2, username):
        dir = '/app/codalab/static/img/partpicture/'+username+'/'
        if os.path.isfile(dir + 'compare.png'):
            os.remove(dir + 'compare.png')
        # plt.plot(query_num, E_in1, 'b', label='AL_train')
        # plt.plot(query_num, E_in2, 'r', label='Random_train')
        plt.plot(query_num, E_out1, 'r', label='AL')
        plt.plot(query_num, E_out2, 'b', label='Random')
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
            qs = UncertaintySampling(trn_ds, method='sm',model=SVM(decision_function_shape='ovr'))
            qs_fordraw = UncertaintySampling(none_trn_ds, method='sm',model=SVM(decision_function_shape='ovr'))
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
        else:
            pass

        return qs, qs_fordraw

    def DeepActiveLearning(self, algorithm, trn_ds, tst_ds, val_ds, lbr, quota, batchsize, vocab_dir, wordslength, numclass, categories_class):
        if algorithm == 'cnn':
            modelcnn = CNN_Probability_Model(vocab_dir, wordslength, batchsize, numclass, categories_class)
            modelcnn.train(trn_ds, val_ds)
            test_acc = modelcnn.test(val_ds)
            E_out = self.realrun_dal(trn_ds, tst_ds, val_ds, lbr, modelcnn, quota, test_acc, batchsize)
        elif algorithm == 'rnn':
            modelrnn = RNN_Probability_Model(vocab_dir, wordslength, batchsize, numclass, categories_class)
            modelrnn.train(trn_ds, val_ds)
            # test_acc = 0.5
            test_acc = modelrnn.test(val_ds)
            E_out = self.realrun_dal(trn_ds, tst_ds, val_ds, lbr, modelrnn, quota, test_acc, batchsize)
        else:
            pass

        return E_out

    def maintodo(self, kind, modelselect, strategy, algorithm, quota, trainAndtest, batchsize, pushallask,docfile,username,useremail,bmtpassword):
        zipfile.ZipFile(docfile).extractall('/app/codalab/thirdpart/'+username)
        train_dir = '/app/codalab/thirdpart/' + username + '/train.txt'
        unlabel_dir = '/app/codalab/thirdpart/' + username + '/unlabel.txt'
        test_dir = '/app/codalab/thirdpart/' + username + '/test.txt'
        vocab_dir = '/app/codalab/thirdpart/' + username + '/vocab/vocab.txt'
        if not os.path.exists('/app/codalab/thirdpart/'+ username + '/vocab/'):
            os.makedirs('/app/codalab/thirdpart/'+ username + '/vocab/')

        # 提交未标记集的两种模式，一种有unlabel文件夹，一种有unlabel文件
        unlabeldatasetdir = '/app/codalab/thirdpart/' + username + '/unlabel/'

        # 将最后问询的结果以csv的格式返回
        willlabel_csvdir = '/app/codalab/static/img/partpicture/' + username + '/dict.csv'
        config = TRNNConfig()

        batchsize = batchsize
        wordslength = config.seq_length
        vocab_size = config.vocab_size
        numclass = config.num_classes


        # [Todo]:需要标记的个数,是否提交的是训练集+测试集，如果只提交一个测试集，那么划分训练集的比例，
        #1是提交的两个文件，一个训练集一个测试集。
        # trainAndtest = 1
        # testsize = 0.33
        # 1是一次提供所有askid, 0是交互式问询
        # pushallask = 1
        #如果是交互式问询的话，那么需要标记的标记列表
        #interlabel = ['0','1']


        # Todo:ask_id是问的train+unlabel中unlabel的id
        maxinitnum = 5 # 为跑图像所允许的每一类别最大的初始标记个数
        unlabeldict = {} # 用作Unlabel的映射关系
        asknamelist = [] # 所有被问询的Unlabel
        E_in1, E_in2 = [], []
        E_out1, E_out2 = [], []
        if os.path.exists(unlabeldatasetdir):
            unlabeldatasetdir = os.listdir(unlabeldatasetdir)
        else:
            unlabeldatasetdir = []

        #提交的是Train还是Train+Test
        if trainAndtest == 1:
            trn_ds, tst_ds, unlabelcontents, unlabelnames, trn_ds_fordraw_fully, trn_ds_fordraw_none, tst_ds_fordraw, quota_fordraw = self.split_train_test_tal(train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, maxinitnum, wordslength)
            # trn_ds_fordraw, n_labeled_fordraw, quota = self.getlabelnum(train_dir, test_dir)
        # 暂时取消提交只有一个Train的逻辑，强行规定必须划分Test哪怕随机划分也好
        #     a =  len(trn_ds.get_labeled_entries()) # 89
        #     b =  len(tst_ds.get_labeled_entries()) # 90
        #     c =  len(trn_ds_fordraw_fully.get_labeled_entries()) # 143
        #     d =  len(trn_ds_fordraw_none.get_labeled_entries()) # 45
        #     e = len(tst_ds_fordraw.get_labeled_entries()) # 36
        #     f = quota_fordraw #98
        #     print cccc
        else:
            pass
            # 提交的只有一个Train
            # X_train, y_train, fully_tst_ds, tst_ds, val_ds = self.split_train_test(trainentity, testsize)
            # trn_ds, numoftrain, unlabelnames = self.split_train_test_unlabel(X_train, y_train, unlabelentity)
            # real_trn_ds = copy.deepcopy(trn_ds)
            # none_trn_ds = self.split_for_drawplot(real_trn_ds, numoftrain, quota)

        trn_ds_random = copy.deepcopy(trn_ds_fordraw_none) # 原验证集强行划分部分未知None做效果用
        qs_random = RandomSampling(trn_ds_random)
        lbr = IdealLabeler(trn_ds_fordraw_fully)

        if strategy == 'binary':
            if modelselect == 'logic':
                qs, qs_fordraw = self.myRegression(algorithm, trn_ds, trn_ds_fordraw_none)
                model = LogisticRegression()
                E_in1, E_out1 = self.realrun_qs(trn_ds_fordraw_none, tst_ds_fordraw, lbr, model, qs_fordraw, quota_fordraw, batchsize)
                E_in2, E_out2 = self.realrun_random(trn_ds_random, tst_ds_fordraw, lbr, model, qs_random, quota_fordraw, batchsize)
            elif modelselect == 'svm':
                qs, qs_fordraw = self.svmClassify(algorithm, trn_ds, trn_ds_fordraw_none)
                model = SVM(kernel='rbf', decision_function_shape='ovr')
                E_in1, E_out1 = self.realrun_qs(trn_ds_fordraw_none, tst_ds_fordraw, lbr, model, qs_fordraw, quota_fordraw, batchsize)
                E_in2, E_out2 = self.realrun_random(trn_ds_random, tst_ds_fordraw, lbr, model, qs_random, quota_fordraw, batchsize)
            elif modelselect == 'dal':
                trn_ds, tst_ds, val_ds, unlabelcontents, unlabelnames, \
                trn_ds_fordraw_fully_dal, trn_ds_fordraw_none_dal, tst_ds_fordraw_dal, val_ds_fordraw, \
                trn_ds_fordraw_fully_tal, trn_ds_fordraw_none_tal, tst_ds_fordraw_tal, \
                quota_fordraw, categories_class, numclass = self.split_train_test_rnn(train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, wordslength)
                lbr_dal = IdealLabeler(trn_ds_fordraw_fully_dal)
                lbr_tal = IdealLabeler(trn_ds_fordraw_fully_tal)
                # trn_ds_random = copy.deepcopy(trn_ds_fordraw_none)  # 原验证集强行划分部分未知None做效果用
                qs_random = RandomSampling(trn_ds_fordraw_none_tal)
                E_out1 = self.DeepActiveLearning(algorithm, trn_ds_fordraw_none_dal, tst_ds_fordraw_dal, val_ds_fordraw, lbr_dal, quota_fordraw, batchsize, vocab_dir, wordslength, numclass, categories_class)
                model = SVM(kernel='rbf', decision_function_shape='ovr')
                E_in2, E_out2 = self.realrun_random(trn_ds_fordraw_none_tal, tst_ds_fordraw_tal, lbr_tal, model, qs_random, quota_fordraw, batchsize)
            else:
                pass

        elif strategy == 'multiclass':
            if modelselect == 'svm':
                qs, qs_fordraw = self.svmClassify(algorithm, trn_ds, trn_ds_fordraw_none)
                model = SVM(kernel='rbf', decision_function_shape='ovr')
                # E_in1, E_out1 = self.score_ideal(trn_ds_fordraw_none, tst_ds_fordraw, lbr, model, qs_fordraw, quota_fordraw)
                E_in1, E_out1 = self.realrun_qs(trn_ds_fordraw_none, tst_ds_fordraw, lbr, model, qs_fordraw, quota_fordraw, batchsize)
                E_in2, E_out2 = self.realrun_random(trn_ds_random, tst_ds_fordraw, lbr, model, qs_random, quota_fordraw, batchsize)
                # E_in2, E_out2 = self.score_ideal(trn_ds_random, tst_ds_fordraw, lbr, model, qs_random, quota_fordraw)
            elif modelselect == 'dal':
                # DAL的部分相对复杂，用两个split函数，一个跑图像（与TAL对比）；另一个拼接（Train和Test）
                trn_ds, tst_ds, val_ds, unlabelcontents, unlabelnames, \
                trn_ds_fordraw_fully_dal, trn_ds_fordraw_none_dal, tst_ds_fordraw_dal, val_ds_fordraw, \
                trn_ds_fordraw_fully_tal, trn_ds_fordraw_none_tal, tst_ds_fordraw_tal, \
                quota_fordraw, categories_class, numclass = self.split_train_test_rnn(train_dir, test_dir, unlabel_dir, vocab_dir, vocab_size, wordslength)
                lbr_dal = IdealLabeler(trn_ds_fordraw_fully_dal)
                lbr_tal = IdealLabeler(trn_ds_fordraw_fully_tal)
                # trn_ds_random = copy.deepcopy(trn_ds_fordraw_none)  # 原验证集强行划分部分未知None做效果用
                qs_random = RandomSampling(trn_ds_fordraw_none_tal)
                E_out1 = self.DeepActiveLearning(algorithm, trn_ds_fordraw_none_dal, tst_ds_fordraw_dal, val_ds_fordraw, lbr_dal, quota_fordraw, batchsize, vocab_dir, wordslength, numclass, categories_class)
                model = SVM(kernel='rbf', decision_function_shape='ovr')
                E_in2, E_out2 = self.realrun_random(trn_ds_fordraw_none_tal, tst_ds_fordraw_tal, lbr_tal, model, qs_random, quota_fordraw, batchsize)
            else:
                pass

        elif strategy == 'multilabel':
            pass

        else:
            pass
        if quota_fordraw % batchsize == 0:
            intern = int(quota_fordraw / batchsize)
        else:
            intern = int(quota_fordraw / batchsize) + 1
        # try:
        self.plotforimage(np.arange(1, intern + 1), E_in1, E_in2, E_out1, E_out2, username)
        # except:
        #     self.plotforimage(np.arange(1, intern + 2), E_in1, E_in2, E_out1, E_out2, username)
        # 返回一批实例,返回分数是为了解决不标注的情况下无法自动更新的问题




        if pushallask == 1:
            if modelselect == 'dal':
                self.DeepActiveLearning(algorithm, trn_ds, tst_ds, val_ds, 'realrun', quota, batchsize, vocab_dir,
                                        wordslength, numclass, categories_class)
            else:
                first, scores = qs.make_query(return_score = True)
                number, num_score = zip(*scores)[0], zip(*scores)[1]
                num_score_array = np.array(num_score)
                max_n = heapq.nlargest(quota, range(len(num_score_array)), num_score_array.take)

            # 只返回文件名
            if len(unlabeldatasetdir) < 1:
                with codecs.open(willlabel_csvdir, 'wb', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['name','entity'])
                    for ask_id in max_n:
                        asknamelist.append(unlabelnames[ask_id])
                        writer.writerow([unlabelnames[ask_id], unlabelcontents[ask_id]])
                return asknamelist, willlabel_csvdir

                # for ask_id in max_n:
                #     filename = unlabelnames[ask_id]
                #     askidlist.append(filename)
                # csvdir = '/app/codalab/static/img/partpicture/' + username + '/dict.csv'
                # with open(csvdir, 'wb') as csv_file:
                #     writer = csv.writer(csv_file)
                #     writer.writerow(['name'])
                #     for unlabelname in askidlist:
                #         writer.writerow([unlabelname])
                # return askidlist, csvdir

            # 如果提交的是一个未标注的文件夹
            else:
                for ask_id in max_n:
                    filename = unlabelnames[ask_id]
                    if filename.split('/')[-1] in unlabeldatasetdir:
                        filenamefull = '/app/codalab/thirdpart/'+username+'/unlabel/'+filename.split('/')[-1]
                        with open(filenamefull) as f:
                            filebody = f.read()
                            unlabeldict[filename] = filebody

                    asknamelist.append(filename)

                csvdir = '/app/codalab/static/img/partpicture/'+username+'/dict.csv'
                with open(willlabel_csvdir, 'wb') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['name', 'entity'])
                    for key, value in unlabeldict.items():
                        #askidlist.append(key)
                        writer.writerow([key, value])
                return asknamelist,willlabel_csvdir

        # 向标注平台发送 [TODO]需要和标注平台融合
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

                asknamelist.append(filename)

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
            return rec_status, rec_url, asknamelist
