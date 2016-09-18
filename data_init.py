"""
Splits the training data into a training (80%), cross-validation (10%) and test (10%) sets.

Usage
    python data_init.py [-h] [input file] [output dir]

for example
    python data_init.py original_data/train.csv data

This creates 3 files (from "original_data/train.csv"):
    * data/train.csv
    * data/cv.csv
    * data/test.csv
"""

import os
import sys
import random

import numpy as np
import pandas as pd


random.seed(0)
np.random.seed(0)


def shuffle(df):
    return df.reindex(np.random.permutation(df.index))


def shuffle_by_label(df, label):
    return shuffle(df.loc[df['Prediction'] == label])


def load_training_data(fname='data/train.csv'):
    # load data frame from the CSV file
    data = pd.read_csv(fname)

    # split into train, cv and test sets
    splitted = (
        shuffle_by_label(data, 1),
        shuffle_by_label(data, 2),
        shuffle_by_label(data, 3),
        shuffle_by_label(data, 4),
        shuffle_by_label(data, 5)
    )

    l = min(map(len, splitted))
    splitted = [s[:l] for s in splitted]

    def sample(dfs, percent):
        r1, r2 = [], []
        for df in dfs:
            i = int(len(df) * percent)
            r1.append(df.iloc[:i])
            r2.append(df.iloc[i:])
        return r1, r2

    # train set with 80% of the data
    train, rest = sample(splitted, 0.8)

    # 10% for the test and cross-validation sets
    cv, test = sample(rest, 0.5)

    def merge(dfs):
        df = pd.concat(dfs)
        return shuffle(df)

    return merge(train), merge(cv), merge(test)


if __name__ == '__main__':
    if '-h' in sys.argv:
        print __doc__
        exit(0)

    if len(sys.argv) != 3:
        print 'Usage: python data_init.py [-h] [input file] [output dir]'
        exit(1)

    train, cv, test = load_training_data(sys.argv[1])

    def fname(name):
        return os.path.join(sys.argv[2], '%s.csv' % name)

    train.to_csv(fname(name='train'))
    cv.to_csv(fname(name='cv'))
    test.to_csv(fname(name='test'))
