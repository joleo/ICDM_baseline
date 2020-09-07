# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       LIYANG099
   date：          2020-8-19
-------------------------------------------------
   Change Activity:
                   2020-8-19:
-------------------------------------------------
"""
__author__ = 'LIYANG099'
import json
import numpy as np
import pandas as pd
import gc
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from keras.utils import multi_gpu_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
from keras.layers import *
from keras.callbacks import *
import pylcs
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

maxlen = 512
epochs = 5
batch_size = 2
learning_rate = 5e-5
min_learning_rate = 1e-5
crf_lr_multiplier = 88  # 50  # 必要时扩大CRF层的学习率
train_flag = 1  # 默认单卡训练

# bert配置
flag = 1
if flag == 1:
    path_file = '/data/data01/liyang099/com/weight/english/cased_L-24_H-1024_A-16/'
    config_path = path_file + 'bert_config.json'
    checkpoint_path = path_file + 'bert_model.ckpt'
    dict_path = path_file + 'vocab.txt'
elif flag == 2:
    path_file = '/data/data01/liyang099/com/weight/english/electra_small/'
    config_path = path_file + 'electra_config.json'
    checkpoint_path = path_file + 'electra_small.ckpt'
    dict_path = path_file + 'vocab.txt'
elif flag == 3:
    path_file = '/data/data01/liyang099/com/weight/english/electra_base/'
    config_path = path_file + 'electra_config.json'
    checkpoint_path = path_file + "electra_base"
    dict_path = path_file + 'vocab.txt'
elif flag == 4:
    path_file = '/data/data01/liyang099/com/weight/english/electra_large/'
    config_path = path_file + 'electra_config.json'
    checkpoint_path = path_file + "electra_large"
    dict_path = path_file + 'vocab.txt'


def get_content(x, y):
    if len(x) > 512:
        if y < 10:
            x = x[0:y + 512][:510]
        elif y > 50:
            x = x[y - 110:y + 460][:510]
        elif y > 30:
            x = x[y - 50:y + 490][:510]
        return x
    else:
        return x


def clean_train(train):
    con = train['content'].apply(len)
    con1 = con < maxlen + 1
    con2 = con > 2
    train = train[con2 & con1].copy()
    train['flag'] = list(map(lambda x, y: 1 if y in x else 0, train['content'], train['entity']))
    train = train[train['flag'] == 1].copy()
    return train


def get_data_new(train):
    train['content'] = list(map(lambda x, y: get_content(x, y), train['content'], train['start']))

    train_data = list()
    for i in train.iloc[:].itertuples():
        train_data.append((i.content, (i.entity, i.label)))
    df = pd.DataFrame(train_data)
    df.columns = ['context', 'entity_list']
    df = df.groupby('context')['entity_list'].apply(list).reset_index()
    df['entity_list'] = df['entity_list'].apply(lambda x: {i[0]: i[1] for i in x})
    train_data = [tuple(i) for i in list(df.values)]
    return train_data


def get_all_data(dataset):
    tmp3 = []
    for i in range(dataset.shape[0]):
        #     start, text, types = [], [], []
        tmp2 = []
        for j in range(len(dataset['reason'][i])):
            #         print(i, j)
            tmp = []
            ids = dataset['id'][i]
            context = dataset['context'][i]
            product_brand = dataset['product/brand'][i]
            start = dataset['reason'][i][j]['start']
            text = dataset['reason'][i][j]['text']
            types = dataset['reason'][i][j]['type']
            res = [ids, context, product_brand, start, text, types]
            tmp.append(res)
            tmp2.extend(tmp)
            tmp3.extend(tmp2)
    df = pd.DataFrame(tmp3, columns={'id', 'content', 'product/brand', 'start', 'entity', 'label'})
    df = df.drop_duplicates(subset=['id', 'content', 'product/brand', 'start', 'entity', 'label']).reset_index(
        drop=True)
    df.columns = ['id', 'content', 'product/brand', 'start', 'entity', 'label']
    return df


train = pd.read_json('./data/train_label_0808.json')
# train = pd.read_json('./data/train_labeled-0808.json')
# train = pd.concat([train1, train2])
train = get_all_data(train)
print(train.shape)
labels = train.label.unique()
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1

te = pd.read_json('./data/test.json')

# train, test = get_data()
# labels = train.label.unique()
# id2label = dict(enumerate(labels))
# label2id = {j: i for i, j in id2label.items()}
# num_labels = len(labels) * 2 + 1
# # train = train.values
# train = get_data_new(train)


te1 = te.copy()
te2 = te.copy()
te3 = te.copy()
te4 = te.copy()
te5 = te.copy()
te6 = te.copy()
te7 = te.copy()
te8 = te.copy()

te1['context'] = te['context'].apply(lambda x: x[0:256])
te2['context'] = te['context'].apply(lambda x: x[256:512])
te3['context'] = te['context'].apply(lambda x: x[512:512 + 256])
te4['context'] = te['context'].apply(lambda x: x[512 + 256:1024])
te5['context'] = te['context'].apply(lambda x: x[1024:1024 + 256])
te6['context'] = te['context'].apply(lambda x: x[1024 + 256:512 + 1024])
te7['context'] = te['context'].apply(lambda x: x[128:512])
te8['context'] = te['context'].apply(lambda x: x[64:64 + 256])
test = pd.concat([te1, te2, te3, te4, te5, te6, te7, te8], axis=0)
test = test[test['context'] != ''].reset_index(drop=True)

te1 = te.copy()
te2 = te.copy()
te3 = te.copy()
te4 = te.copy()
te5 = te.copy()
te6 = te.copy()

te1['context'] = te['context'].apply(lambda x: x[0:512])
te2['context'] = te['context'].apply(lambda x: x[256:512 + 256])
te3['context'] = te['context'].apply(lambda x: x[512:1024])
te4['context'] = te['context'].apply(lambda x: x[128:512 + 128])
te5['context'] = te['context'].apply(lambda x: x[64:512 + 64])
te6['context'] = te['context'].apply(lambda x: x[0:256])

test = pd.concat([te1, te2, te3, te4, te5], axis=0)
test = test[test['context'] != ''].reset_index(drop=True)
test[:2]

train = get_data_new(train)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=512)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = tokenizer.encode(argument[0])[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[argument[1]] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def viterbi_decode(nodes, trans):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[:, 0].argmax()]


def extract_arguments(text, model, CRF):
    """ arguments抽取函数 """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 510:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(CRF.trans)
    labels = viterbi_decode(nodes, trans)
    arguments, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[i], id2label[(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    try:
        return {
            text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1]: l
            for w, l in arguments
        }
    except:
        return {}


def evaluate(data, model, CRF):
    """评测函数（跟官方评测结果不一定相同，但很接近）"""
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments(text, model, CRF)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall



class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """

    def __init__(self, valid_data, model, CRF, file_path):
        self.best_val_f1 = 0.
        self.valid_data = valid_data
        self.model = model
        self.CRF = CRF
        self.passed = 0
        self.file_path = file_path

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate(self.valid_data, self.model, self.CRF)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights(self.file_path)
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

    def evaluate(self, data, model, CRF):
        """评测函数（跟官方评测结果不一定相同，但很接近）
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for text, arguments in tqdm(data):
            inv_arguments = {v: k for k, v in arguments.items()}
            pred_arguments = extract_arguments(text, model, CRF)
            pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
            Y += len(pred_inv_arguments)
            Z += len(inv_arguments)
            for k, v in pred_inv_arguments.items():
                if k in inv_arguments:
                    # 用最长公共子串作为匹配程度度量
                    l = pylcs.lcs(v, inv_arguments[k])
                    X += 2. * l / (len(v) + len(inv_arguments[k]))
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall


def build_bert(num_labels):
    model = build_transformer_model(config_path, checkpoint_path)  # ,model = 'electra')
    for layer in model.layers:
        layer.trainable = True
    # bilstm = Bidirectional(GRU(200, return_sequences=True))(model.output)
    #     bilstm = SpatialDropout1D(0.5)(bilstm)
    output = Dense(num_labels)(model.output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)
    # model.summary()

    # model = multi_gpu_model(model, gpus= 2)
    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learning_rate),
        metrics=[CRF.sparse_accuracy]
    )
    return model, CRF


def run_cv(nfolds, data, data_label, data_test, data_valid, epochs=10, date_str='1107'):
    skf = KFold(n_splits=nfolds, shuffle=True, random_state=48).split(train)
    #     train_model_pred = np.zeros((len(data), n_class))
    #     test_model_pred = np.zeros((len(data_test), n_class))

    for k, (train_fold, test_fold) in enumerate(skf):
        print('Fold: ', k)
        if k != 10:
            #         if k in [0,1,2]:

            '''数据部分'''
            # 数据划分
            train_data, valid_data, = [tuple(i) for i in list(np.array(train)[train_fold])], [tuple(i) for i in list(
                np.array(train)[test_fold])]

            '''模型部分'''
            # 生成模型
            model, CRF = build_bert(num_labels)
            file_path = date_str + str(k) + '.weights'

            evaluator = Evaluator(valid_data, model, CRF, file_path)
            if not os.path.exists(file_path):
                train_generator = data_generator(train_data, batch_size)
                valid_generator = data_generator(valid_data, batch_size)

                model.fit_generator(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    epochs=epochs,
                    verbose=1,
                    callbacks=[evaluator])
                model.load_weights(file_path)
            else:
                model.load_weights(file_path)
            data_test['submit_' + str(k)] = data_test['context'].apply(lambda x: extract_arguments(x, model, CRF))
            print(data_test['submit_' + str(k)])
            print('Fold: ', sum(data_test['submit_' + str(k)].apply(len) > 0))
            #         data_valid['submit_'+str(k)] = data_valid['content'].apply(lambda x:extract_arguments(x,model,CRF))
            del model
            del CRF
            gc.collect()
            K.clear_session()
        else:
            continue
    return data_test, data_valid


data_test1, data_valid1 = run_cv(10, train, None, test, None, epochs=10, date_str='./model/bert_large/')

# 测试0.348
data_test1, data_valid1 = run_cv(30, train, None, test, None, epochs=10, date_str='./model/electra_large_30_fold_0812/')

# electrac base 30 flods
data_test1, data_valid1 = run_cv(35, train, None, test, None, epochs=10, date_str='./model/electra_base_35_fold_0812/')

from collections import Counter


def counter(arr):
    return Counter(arr).most_common(1)


def drop_dup(x):
    set_list = list()
    for i in x:
        if i not in set_list:
            set_list.append(i)
    return set_list


def get_top2(x):
    all_list = list()
    #     if len(x)
    for i in x:
        if i[1] > 0:
            all_list.append(i[0])
    return [list(i) for i in all_list]


def recommendation_user_list(result):
    z = result.groupby(['content'])['all_key_2'].apply(lambda x: np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['content'], row['all_key_2']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['items', 'content']
    return i


def recommendation_user_list(result):
    z = result.groupby(['id'])['all_key_2'].apply(lambda x: np.concatenate(list(x))).reset_index()
    i = pd.concat([pd.Series(row['id'], row['all_key_2']) for _, row in z.iterrows()]).reset_index()
    i.columns = ['items', 'uid']
    return i


def submit(data_test):
    sub = data_test.copy()
    for i in range(10):
        sub['key_' + str(i)] = sub['submit_' + str(i)].apply(lambda x: [i for i in list(x.items())])
        if i == 0:
            sub['all_key'] = sub['key_0']
        else:
            sub['all_key'] = sub['all_key'] + sub['key_' + str(i)]
    sub['all_key_1'] = sub['all_key'].apply(drop_dup)
    sub = sub.groupby('id')['all_key'].apply(lambda x: list(x)[0]).reset_index()
    sub['all_key_2'] = sub['all_key'].apply(counter)
    sub['all_key_2'] = sub['all_key_2'].apply(get_top2)
    submit_1 = sub[['id', 'all_key_2']].copy()
    submit_1 = recommendation_user_list(submit_1)
    submit_1['entity'] = submit_1['items'].apply(lambda x: x[0])
    submit_1['label'] = submit_1['items'].apply(lambda x: x[1])
    submit_1 = submit_1[['uid', 'label', 'entity']]
    submit_1 = submit_1.drop_duplicates()
    return submit_1


sub1 = submit(data_test1)
sub1.columns = ['id', 'type', 'text']
sub1.shape


def get_data():
    train = pd.read_json('./data/train.json')

    train['start'] = train['reason'].apply(lambda x: x[0]['start'])
    train['text'] = train['reason'].apply(lambda x: x[0]['text'])
    train['type'] = train['reason'].apply(lambda x: x[0]['type'])
    test = pd.read_json('./data/test.json')
    return train, test


tr, Te = get_data()

te = Te.merge(sub1, on='id', how='left')
te = te.fillna('')
te['start'] = list(map(lambda x, y: x.index(y), te['context'], te['text']))
te['reason'] = list(map(lambda x, y, z: [{'start': x, 'text': y, 'type': z}], te['start'], te['text'], te['type']))
train = pd.read_json('./data/train.json')
te = te[train.columns]
te.to_json('./submit/result.json', orient='records', force_ascii=False)