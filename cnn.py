#!/usr/bin/python3

from tf.keras.models import Model
from tf.keras.layers import Dense,Conv2D,Activation,MaxPooling2D,GlobalAveragePooling2D
import numpy as np
import tensorflow as tf

ASC_CLASS=5

'''
    4層 畳み込みニューラルネットワーク
'''
class CNN4(object):

    def __init__(self):
        # モデルパラメータ
        self.filters=64
        self.max_depth=4
        self.kernel_size=3
        self.pool_size=2

    def __call__(self, x):

    # 4層分繰り返す
    for depth in range(self.max_depth):
        x=Conv2D(filters=self.filters*(depth+1)
                kernel_size=self.kernel_size, padding='same',
                data_format='channels_last',
                kernel_initializer='glorot_uniform')(x)
        x=Activation('relu')(x)
        # 最大値プーリング
        x=MaxPooling2D(pool_size=(self.pool_size,self.pool_size),
                data_format='channels_last')(x)

    # 特徴量，時間方向に平均プーリング
    x=GlobalAveragePooling2D(data_format='channels_last')(x)

    # 確率ベクトルを出力
    output=Dense(units=ASC_CLASS, kernel_initializer='uniform', activation='softmax')(x)

    # 5個の確率がおさまったベクトルを返す
    return output
