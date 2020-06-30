#!/usr/bin/python3

import os
import numpy as np
import scipy
import h5py
import re
import csv
import librosa
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    with h5py.File(args.data, 'r') as data:
        keys = data.keys()
        print("size=%d" % len(keys))

if __name__ == "__main__":
  main()
