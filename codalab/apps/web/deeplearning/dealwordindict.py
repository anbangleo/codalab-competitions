# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr
from sklearn import preprocessing
import jieba
from libact.base.dataset import Dataset, import_libsvm_sparse

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

# class Dealword():
def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    content = list(jieba.cut(native_content(content)))
                    contents.append(content)
                    #contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

def read_file_nocut(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(native_content(content))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

# 可变参数为可能的【未标注集】
def build_vocab(train_dir, vocab_dir, vocab_size=1000, *args):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    print ("start to build vob....")
    all_data = []
    # print (data_train)

    for content in data_train:
        all_data.extend(content)
    # 提交的有一个未标记集
    if len(args) == 1:
        none_dataset = args[0]
        unlabel_data_train, unlabel_cat = read_file(none_dataset)
        for content in unlabel_data_train:
            all_data.extend(content)
    # 提交的是未标记集 + 测试集
    elif len(args) == 2:
        none_dataset = args[0]
        test_dataset = args[1]
        unlabel_data_train, unlabel_cat = read_file(none_dataset)
        for content in unlabel_data_train:
            all_data.extend(content)
        test_data, test_cat = read_file(test_dataset)
        for content in test_data:
            all_data.extend(content)
    else:
        pass

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    print ("Finish building vocab!")

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

# 读train_dir以获取输入文件中所有的样本类别
def read_category(filename):
    """读取分类目录，固定"""
    categories = []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if label not in categories:
                    categories.append(label)
            except:
                pass
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

# def process_al_file(data, word_to_id,cat_to_id,max_length=600):
#     labels,contents = zip(*dataset.format_sklearn())
#     data_id, label_id = [], []
#set
#
#     for line in contents:
#         try:
#             if content:
#                     content = list(jieba.cut(native_content(content)))
#                     contents.append(content)
#                     #contents.append(list(native_content(content)))
#                     labels.append(native_content(label))
#             except:
#                 pass
#     # return contents, labels
#
#     for i in range(len(contents)):
#         data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])  ##把contents中的每一个词用word_to_id中id表示
#         label_id.append(cat_to_id[labels[i]])
#
#     x_pad = []
#     res = []
#     two = []
#     for i in data_id:
#         ll = len(i)
#         rank = 0
#         q = 0
#         for j in range(600):
#             a = i.count(j)
#             if a > 0:
#                 res.append(a)
#             else:
#                 res.append(0)
#         x_pad.append(res)
#         res = []
#
#     x_pad = np.array(x_pad)
#     y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
#     print("Finish dealing with files!")
#     return x_pad, y_pad


def process_file(filename, word_to_id, cat_to_id, unlabelflag, max_length=600):
    """将文件转换为id表示"""
    print ("Start to deal with file...")
    contents, labels = read_file(filename)
    # for i in contents[0]:
        # print i.encode('utf-8')
    data_id, label_id = [], []
    #for t in contents:
    #    t = list(jieba.cut(t))
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])##把contents中的每一个词用word_to_id中id表示
        # print data_id
        # break
        label_id.append(cat_to_id[labels[i]])

    x_pad = []
    res = []
    for i in data_id:

        for j in range(max_length):
    #        # a = format(float(i.count(j))/float(ll),'.6f')
            a = i.count(j)
            if a > 0:
                res.append(a)
            else:
                res.append(0)
        x_pad.append(res)
        res=[]

  #  x_pad = np.array(x_pad)
  #  y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    print ("Finish dealing with files!")

    if unlabelflag == 1:
        return x_pad, label_id, contents, labels
    else:
        return x_pad, label_id

# labels 是所有Label 的list，对未标记集需要提供（未标记的名字）
def process_file_rnn(filename, word_to_id, cat_to_id, unlabelflag, max_length=600):
    """将文件转换为id表示"""
    print ("Start to deal with file...")
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])##把contents中的每一个词用word_to_id中id表示
        if unlabelflag == 1: # 对Unlabel处理
            label_id.append(-1)
        else:
            label_id.append(cat_to_id[labels[i]])

    # x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    print ("Finish dealing with files!")
    if unlabelflag == 1:
        contents, labels = read_file_nocut(filename)
        return data_id, label_id, contents, labels
    else:
        return data_id, label_id

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
