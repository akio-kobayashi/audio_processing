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

'''
    ニューラルネットワークの学習と評価に使う関数
    ノートブックからよびだして使う
'''
def train(hires=False, filters=16, max_depth=2, kernel_size=2, pool_size=2, doubling=False, algorithm='adam'):

    '''
        *** 学習時のパラメータを設定 ***
    '''
    input_length=552
    batch_size=10
    epochs=10
    learn_rate=1.0e-3

    '''
        *** 学習と評価用のデータを選ぶ ***
    '''
    if hires is True:
        # 80次元のデータ．こちらの方が詳細なデータ
        input_dim=80
        train_data='train_hires.h5'
        test_data='test_hires.h5'
    else:
        # 40次元のデータ．デフォルト
        input_dim=40
        train_data='./train.h5'
        test_data='./test.h5'

    '''
        *** 学習に使うアルゴリズム（最適化手法）の選択 ***
    '''
    if algorithm == 'sgd':
        optimizer=tf.keras.optimizers.SGD()
    elif algorithm == 'adadelta':
        optimizer=tf.keras.optimizers.Adadelta()
    elif algorithm == 'rmsprop':
        optimizer=tf.keras.optimizers.RMSprop()
    else:
        optimizer=tf.keras.optimizers.Adam()

    '''
        *** 入力ベクトルの形状(次元)を指定 ***
        ( 特徴量の次元，時間(長さ)の次元，フィルタ数の次元 ) = numpyのshape
    '''
    input=tf.keras.layers.Input( shape=(input_dim, input_length, 1) )

    '''
        *** ニューラルネットワークを作成 ***
        どのようにして作っているかはcnn.pyを読むこと
    '''
    output=CNNClassifier()(input)

    '''
        *** モデルの設定 ***
        ニューラルネットワークの入出力を指定し，学習・評価できるように設定する
    '''
    model=tf.keras.models.Model(input, output)
    # 学習可能な計算グラフを作成
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    '''
        *** 学習データの準備 ***
    '''
    training_generator = DataGenerator(train_data, dim=(input_dim, input_length),
                                          batch_size=batch_size)
    # 入力データが正規分布にしたがうよう，偏りをなくす（正規化）
    training_generator.compute_norm()
    mean, var=training_generator.get_norm()

    '''
        *** 評価データの準備 ***
    '''
    validation_generator = DataGenerator(test_data, dim=(input_dim, input_length),
                                            batch_size=batch_size)
    validation_generator.set_norm(mean, var)

    '''
        *** Tensorboardを使って学習状況のログを保存する***
    '''
    try:
        shutil.rmtree('./logs')
    except:
        pass
    os.makedirs('./logs')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

    '''
        *** 学習 ***
        tensorflowの以下の関数を呼び出して行う．
    '''
    model.fit_generator(training_generator,
                        validation_data=validation_generator,
                        epochs=epochs,
                        callbacks=[tensorboard],
                        shuffle=True)

    '''
        *** 評価 ***
        学習したニューラルネットワークを使って評価を行う
    '''
    # 混同行列の初期化
    mat=np.zeros((ASC_CLASS,ASC_CLASS), dtype=int)
    # 正解数の初期化
    acc=0

    for bt in range(validation_generator.__len__()):
        # データを10個取り出す
        # x ... メルスペクトル特徴, y...ラベル
        x,y = validation_generator.__getitem__(bt)
        # 学習したモデルで予測する
        # pred は5つの鳴き声の確率が入ったベクトルを10個ならべたもの
        pred = model.predict_on_batch(x)

        # 確率が最大となるラベルの番号を求める
        y_pred=np.argmax(pred, axis=1)
        y_true=np.argmax(y, axis=1)

        # 混同行列を作る
        mat+=confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])

        # 正解と予測ラベルを比較して，正解した数を加算
        acc+=np.sum(y_pred == y_true)

    # 評価データの総数で割る
    acc = float(acc)/validation_generator.__num_samples__()

    return acc,mat
