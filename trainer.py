#!/usr/bin/python3

import os
import sys
import subprocess
import time
import shutil
import numpy as np
from cnn import CNN4
from generator import DataGenerator
#from tf.keras.callbacks import TensorBoard
#from tf.keras.models import Model
#from tf.keras.layers import Input
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

ASC_CLASS=5

def trainer():

    input_dim=40
    input_length=500
    batch_size=10
    epochs=10
    learn_rate=1.0e-3

    # 学習と評価用のデータ
    train_data='train.h5'
    test_data='test.h5'

    # 学習に使うアルゴリズムの選択
    optimizer=tf.keras.optimizers.Adam()

    # 入力ベクトルの形状(次元)を指定
    # ( 特徴量の次元，時間(長さ)の次元，フィルタ数の次元 ) = numpyのshape
    input=tf.keras.layers.Input( shape=(input_dim, input_length, 1) )

    # ニューラルネットワーク
    output=CNN4()(input)

    # モデルの定義
    model=tf.keras.mdoels.Model(input, output)
    # 学習可能な計算グラフを作成
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # データの準備
    training_generator = DataGenerator(train_data, dim=(input_dim, input_length),
                                          batch_size=batch_size)
    # 入力データが正規分布にしたがうよう，偏りをなくす
    training_generator.compute_norm()
    mean, var=training_generator.get_norm()

    # 128個を1つのバッチとしたデータを作成（評価用）
    validation_generator.set_norm(mean, var)
    validation_generator = DataGenerator(test_data, dim=(input_dim, input_length),
                                            batch_size=batch_size)

    # 学習状況のログを保存する
    try:
        shutil.rmtree('./logs')
    except:
        pass
    os.makedirs('./logs')
    # Tensorboardに記録する
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

    # 学習
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=[tensorboard],
                        shuffle=True)

    mat=np.zeros((ASC_CLASS,ASC_CLASS), dtype=int)
    acc=np.zeros((ASC_CLASS,ASC_CLASS), dtype=int)
    for bt in range(valid_generator.__len__()):
        # データを10個取り出す
        x,y = valid_generator.__getitem__(bt)
        # 学習したモデルで予測する
        pred = model.predict_on_batch(x)

        # 確率が最大となるインデックスを求める
        y_pred=np.argmax(pred, axis=1)
        y_true=np.argmax(y_batch, axis=1)

        # 正解率の計算に使う行列を作る
        conf_mat=confusion_matrix(y_true, y_pred)
        mat=np.add(conf_mat, mat)

        acc=np.add(y_pred == y_true)
    acc = np.mean(acc)
    print("正解率: %.4f %" % acc)
    #print("クラスごとの予測")
    #print(convf_mat)
    return acc, conv_mat
