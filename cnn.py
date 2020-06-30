#!/usr/bin/python3

from tf.keras.models import Model
from tf.keras.layers import Dense,Input,Conv2D,Activation,MaxPooling2D,GlobalAveragePooling2D
import numpy as np
import tensorflow as tf

ASC_CLASS=4

'''
    4-layer 畳み込みニューラルネットワーク
'''
class CNN4(object):

    def __init__(self):

    def __call__(self, x):

        for depth in range(4):
            x=Conv2D(filters=64*(depth+1)
                    kernel_size=3, padding='same',
                    data_format='channels_last',
                    kernel_initializer='glorot_uniform')(x)
            x=Activation('relu')(x)
            # 最大値プーリング
            x=MaxPooling2D(pool_size=(2,2), data_format='channels_last')(x)
        # 特徴量，時間方向に平均プーリング
        x=GlobalAveragePooling2D(data_format='channels_last')(x)
        # (256) 次元 -> (10) 次元
        output=Dense(units=ASC_CLASS, kernel_initializer='uniform', activation='softmax')(x)

        return output   # 4個の確率がおさまったベクトル
