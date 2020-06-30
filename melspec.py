#!/usr/bin/python3

import os
import numpy as np
import scipy
import h5py
import re
import csv
import librosa
import argparse

def compute_melspec(signal):
    melspec=librosa.feature.melspectrogram(signal, sr=16000, S=None, n_fft=512, hop_length=200, win_length=400,
                                            window='hamming', center=True, pad_mode='reflect', power=2.0, n_mels=40)
    melspec = np.log(melspec+1.0e-8)

    return melspec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--valid', type=str, required=True)
    args = parser.parse_args()

    with open(args.list) as fp:
        dirlist=fp.readlines()

    data={}
    keys=[]
    with open(args.csv) as fp:
        csv_file = open(args.csv, "r")
        df = csv.DictReader(csv_file)
        for row in df:
            keys.append(row['key'])
            data[row['key']] = row

    for n in range(5):
        counts[n]=0

    with h5py.File(args.valid, 'w') as valid:
        with h5py.File(args.train, 'w') as train:
            for key in keys:
                if data[key]['category'] == 'cat':
                    label=0
                elif data[key]['category'] == 'cow':
                    label=1
                elif data[key]['category'] == 'dog':
                    label=2
                elif data[key]['category'] == 'frog':
                    label=3
                elif data[key]['category'] == 'pig':
                    label=4
                else:
                    continue
                counts[label] += 1

                path=os.path.join('./audio',data[key]['filename'])
                wav,sr=librosa.load(path)
                mels=compute_melspec(wav)
                if counts[label] > 30:
                    valid.create_group(key)
                    valid.create_dataset(key+'/feature', data=mels)
                    valid.create_dataset(key+'/label', data=mels)
                else:
                    train.create_group(key)
                    train.create_dataset(key+'/feature', data=mels)
                    train.create_dataset[key+'/label', data=label]


if __name__ == "__main__":
  main()
