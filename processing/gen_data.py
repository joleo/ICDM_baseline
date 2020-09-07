# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 20:44
@Auth ： joleo
@File ：gen_data.py
"""
import pandas as pd
from config import *


def get_content(x,y):
    if len(x)>512:
         if y<10:
             x = x[0:y+512][:510]
         elif y>50:
             x = x[y-110:y+460][:510]
         elif y>30:
             x = x[y-50:y+490][:510]
         return x
    else: return x


def clean_train(train):
    con = train['content'].apply(len)
    con1 = con<maxlen+1
    con2 = con>2
    train = train[con2&con1].copy()
    train['flag'] = list(map(lambda x,y:1 if y  in x else 0 ,train['content'],train['entity']))
    train = train[train['flag']==1].copy()
    return train


def get_data_new(train):
    train['content'] = list(map(lambda x,y:get_content(x,y),train['content'],train['start']))

    train_data = list()
    for i in train.iloc[:].itertuples():
        train_data.append((i.content,(i.entity,i.label)))
    df = pd.DataFrame(train_data)
    df.columns = ['context','entity_list']
    df = df.groupby('context')['entity_list'].apply(list).reset_index()
    df['entity_list'] = df['entity_list'].apply(lambda x:{i[0]:i[1] for i in x})
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
            types =dataset['reason'][i][j]['type']
            res = [ids, context, product_brand, start, text, types]
            tmp.append(res)
            tmp2.extend(tmp)
            tmp3.extend(tmp2)
    df = pd.DataFrame(tmp3, columns={'id', 'content', 'product/brand', 'start', 'entity', 'label'})
    df = df.drop_duplicates(subset=['id', 'content', 'product/brand', 'start', 'entity', 'label']).reset_index(drop=True)
    df.columns = ['id', 'content', 'product/brand', 'start', 'entity', 'label']
    return df


def gen_test(te):
    te1 = te.copy()
    te2 = te.copy()
    te3 = te.copy()
    te4 = te.copy()
    te5 = te.copy()
    te6 = te.copy()
    te7 = te.copy()
    te8 = te.copy()

    te1['context'] = te['context'].apply(lambda x:x[0:256])
    te2['context'] = te['context'].apply(lambda x:x[256:512])
    te3['context'] = te['context'].apply(lambda x:x[512:512+256])
    te4['context'] = te['context'].apply(lambda x:x[512+256:1024])
    te5['context'] = te['context'].apply(lambda x:x[1024:1024+256])
    te6['context'] = te['context'].apply(lambda x:x[1024+256:512+1024])
    te7['context'] = te['context'].apply(lambda x:x[128:512])
    te8['context'] = te['context'].apply(lambda x:x[64:64+256])
    test = pd.concat([te1,te2,te3,te4,te5,te6,te7,te8],axis= 0 )
    test = test[test['context']!=''].reset_index(drop = True)
    return test


def gen_test_2(te):
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
    return test


def get_data():
    train = pd.read_json('./data/train.json')

    train['start'] = train['reason'].apply(lambda x: x[0]['start'])
    train['text'] = train['reason'].apply(lambda x: x[0]['text'])
    train['type'] = train['reason'].apply(lambda x: x[0]['type'])
    test = pd.read_json('./data/test.json')
    return train, test