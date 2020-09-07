# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 20:43
@Auth ： joleo
@File ：config.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

maxlen = 512
epochs = 5
batch_size = 2
learning_rate = 5e-5
min_learning_rate = 1e-5
crf_lr_multiplier = 88#50
n_flod = 30
epoch = 10
model_name = 'electra'

flag = 1
if flag== 1:
    path_file = '/data/data01/liyang099/com/weight/english/cased_L-24_H-1024_A-16/'
    config_path = path_file +  'bert_config.json'
    checkpoint_path = path_file +  'bert_model.ckpt'
    dict_path = path_file +  'vocab.txt'
elif flag == 2:
    path_file = '/data/data01/liyang099/com/weight/english/electra_small/'
    config_path = path_file +  'electra_config.json'
    checkpoint_path = path_file +  'electra_small.ckpt'
    dict_path = path_file +  'vocab.txt'
elif flag == 3:
    path_file = '/data/data01/liyang099/com/weight/english/electra_base/'
    config_path = path_file +  'electra_config.json'
    checkpoint_path = path_file +  "electra_base"
    dict_path = path_file +  'vocab.txt'
elif flag == 4:
    path_file = '/data/data01/liyang099/com/weight/english/electra_large/'
    config_path = path_file +  'electra_config.json'
    checkpoint_path = path_file +  "electra_large"
    dict_path = path_file +  'vocab.txt'


