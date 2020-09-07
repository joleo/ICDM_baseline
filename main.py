# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 20:45
@Auth ： joleo
@File ：main.py
"""
from config import *
from processing.gen_data import *
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
from sklearn.model_selection import train_test_split, KFold
from model.gen_bert_crf import build_crf_bert
import pickle
import os
from processing.processing_final_result import *

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i: i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):
    """数据生成器 """
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
    """Viterbi算法求最优路径,其中nodes.shape=[seq_len, num_labels],trans.shape=[num_labels, num_labels].
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
    """评测函数"""
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
    """评估和保存模型"""
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


def run_cv(nfolds, train, data_test, model_name, date_str, epochs=10):
    skf = KFold(n_splits=nfolds, shuffle=True, random_state=48).split(train)
    for k, (train_fold, test_fold) in enumerate(skf):
        print('Fold: ', k)
        if k != 10:
            '''数据部分'''
            # 数据划分
            train_data, valid_data, = [tuple(i) for i in list(np.array(train)[train_fold])], [tuple(i) for i in list(
                np.array(train)[test_fold])]

            '''模型部分'''
            # 生成模型
            model, CRF = build_crf_bert(num_labels, model_name)
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
    return data_test


if __name__ == '__main__':
    train = pd.read_json('./data/train_label_0808.json')
    te  = pd.read_json('./data/test.json')
    train = get_all_data(train)
    test = gen_test_2(te)
    print(train.shape)

    tr_col = train.columns
    labels = train.label.unique()
    id2label = dict(enumerate(labels))
    label2id = {j: i for i, j in id2label.items()}
    num_labels = len(labels) * 2 + 1

    train = get_data_new(train)

    data_test1 = run_cv(n_flod=n_flod, train=train, data_test=test, epochs=epoch, model_name=model_name, date_str='./data/weight/bert_large/')

    # data_test1, data_valid1 = run_cv(30, train, None, test, None, epochs=10, model_name='electra', date_str='./data//model/electra_large_30_fold_0812/')
    # electrac base 30 flods
    # data_test1, data_valid1 = run_cv(35, train, None, test, None, epochs=10, model_name='electra', date_str='./data//model/electra_base_35_fold_0812/')

    sub1 = submit(data_test1)
    sub1.columns = ['id', 'type', 'text']
    tr, Te = get_data()
    te = Te.merge(sub1, on='id', how='left')
    # te = te.fillna('')
    te['start'] = list(map(lambda x, y: x.index(y), te['context'], te['text']))
    te['reason'] = list(map(lambda x, y, z: [{'start': x, 'text': y, 'type': z}], te['start'], te['text'], te['type']))

    te = te[tr_col]
    te.to_json('./result/result.json', orient='records', force_ascii=False)