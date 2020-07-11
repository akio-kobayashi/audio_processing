#!/usr/bin/python3

#from tf.keras.models import Model
#from tf.keras.layers import Dense,Conv2D,Activation,MaxPooling2D,GlobalAveragePooling2D
import numpy as np
import tensorflow as tf

ASC_CLASS=5

'''
    4層 畳み込みニューラルネットワーク
'''
class CNNClassifier(object):

    def __init__(self, filters=64, max_depth=4, kernel_size=3, pool_size=2, doubling=False):
        # モデルパラメータ
        self.filters=filters
        self.max_depth=max_depth
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.doubling=doubling

    def __call__(self, x):

        # 4層分繰り返す
        for depth in range(self.max_depth):
            # 出力 = 層を表す関数（入力）と記述する
            # 1. 畳み込み
            x=tf.keras.layers.Conv2D(filters=self.filters*(depth+1),
                    kernel_size=self.kernel_size, padding='same',
                    data_format='channels_last',
                    kernel_initializer='glorot_uniform')(x)
            if self.doubling is True:
                x=tf.keras.layers.Conv2D(filters=self.filters*(depth+1),
                        kernel_size=self.kernel_size, padding='same',
                        data_format='channels_last',
                        kernel_initializer='glorot_uniform')(x)
            # 2. 活性化関数 (Rectified Linear Unit)
            x=tf.keras.layers.Activation('relu')(x)
            # 3. 最大値プーリング
            x=tf.keras.layers.MaxPooling2D(pool_size=(self.pool_size,self.pool_size),
                    data_format='channels_last')(x)

        # 特徴量，時間方向に平均する
        x=tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)

        # 確率ベクトルを出力
        output=tf.keras.layers.Dense(units=ASC_CLASS, kernel_initializer='uniform', activation='softmax')(x)

        # 5個の確率がおさまったベクトルを返す
        return output
