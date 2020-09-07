# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 20:48
@Auth ： joleo
@File ：gen_bert_crf.py
"""
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField
from bert4keras.backend import search_layer, K
from keras.models import Model
from keras.layers import *
from config import *

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


def build_crf_adversarial_bert(num_labels, model_name='electra'):
    model = build_transformer_model(config_path, checkpoint_path,model = model_name)
    for layer in model.layers:
        layer.trainable = True
    output = Dense(num_labels)(model.output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learning_rate),
        metrics=[CRF.sparse_accuracy]
    )

    return model, CRF


def build_crf_bert(num_labels, model_name='electra'):
    model = build_transformer_model(config_path, checkpoint_path,model = model_name)
    for layer in model.layers:
        layer.trainable = True
    output = Dense(num_labels)(model.output)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learning_rate),
        metrics=[CRF.sparse_accuracy]
    )

    adversarial_training(model, 'Embedding-Token', 0.5)
    return model, CRF


def build_lstm_crf_bert(num_labels, model_name):
    model = build_transformer_model(config_path, checkpoint_path, model = model_name)
    for layer in model.layers:
        layer.trainable = True
    bilstm = Bidirectional(GRU(200, return_sequences=True))(model.output)
    bilstm = SpatialDropout1D(0.5)(bilstm)
    output = Dense(num_labels)(bilstm)
    CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
    output = CRF(output)

    model = Model(model.input, output)

    # model = multi_gpu_model(model, gpus= 2)
    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learning_rate),
        metrics=[CRF.sparse_accuracy]
    )
    model.summary()
    return model, CRF
