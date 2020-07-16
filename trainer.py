#!/usr/bin/python3

import os
import sys
import subprocess
import time
import shutil
import numpy as np
from cnn import CNNClassifier
from generator import DataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

ASC_CLASS=5

def train(hires=False, filters=16, max_depth=2, kernel_size=2, pool_size=2, doubling=False, algorigim='adam'):

    input_length=552
    batch_size=10
    epochs=10
    learn_rate=1.0e-3

    # 学習と評価用のデータ
    if hires is True:
        input_dim=80 # 80次元のデータ．こちらの方が詳細なデータ
        train_data='train_hires.h5'
        test_data='test_hires.h5'
    else:
        input_dim=40 # 40次元のデータ．デフォルト
        train_data='./train.h5'
        test_data='./test.h5'

    # 学習に使うアルゴリズムの選択
    if algorithm == 'sgd':
        optimizer=tf.keras.optimizers.SGD()
    elif algorithm == 'adadelta':
        optimizer=tf.keras.optimizers.AdaDelta()
    elif algorigim == 'rmsprop':
        optimizer=tf.keras.optimizers.RMSprop()
    else:
        optimizer=tf.keras.optimizers.Adam()

    # 入力ベクトルの形状(次元)を指定
    # ( 特徴量の次元，時間(長さ)の次元，フィルタ数の次元 ) = numpyのshape
    input=tf.keras.layers.Input( shape=(input_dim, input_length, 1) )

    # ニューラルネットワーク
    output=CNNClassifier()(input)

    # モデルの定義
    model=tf.keras.models.Model(input, output)
    # 学習可能な計算グラフを作成
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # データの準備
    training_generator = DataGenerator(train_data, dim=(input_dim, input_length),
                                          batch_size=batch_size)
    # 入力データが正規分布にしたがうよう，偏りをなくす
    training_generator.compute_norm()
    mean, var=training_generator.get_norm()

    # 10個のデータを1つのバッチとしたデータを作成（評価用）
    validation_generator = DataGenerator(test_data, dim=(input_dim, input_length),
                                            batch_size=batch_size)
    validation_generator.set_norm(mean, var)

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
    acc=0
    num=0
    for bt in range(validation_generator.__len__()):
        # データを10個取り出す
        x,y = validation_generator.__getitem__(bt)
        # 学習したモデルで予測する
        pred = model.predict_on_batch(x)

        # 確率が最大となるインデックスを求める
        y_pred=np.argmax(pred, axis=1)
        y_true=np.argmax(y, axis=1)

        # 正解率の計算に使う行列を作る
        conf_mat=confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
        mat+=conf_mat

        acc+=np.sum(y_pred == y_true)
        num+=len(y_true)
    acc = float(acc)/num
    return acc,mat
