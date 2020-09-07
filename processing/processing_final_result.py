# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 20:51
@Auth ： joleo
@File ：processing_final_result.py
"""
import pandas as pd
import numpy as np
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