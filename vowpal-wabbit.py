# coding: utf-8

"""
Converts the training data into VW format.

Usage
    python vw.py [output dir]

./vw
====
    --oaa k: One Against All, k = nb classes


train
=====
    vw vw/train.vw -c -k --passes 300 --ngram 7 -b 24 --ect 5 -f vw/train.model.vw > /dev/null

test
====
    vw vw/cv.vw -t -i vw/train.model.vw -p cv.preds.txt > /dev/null


vw vw/train.vw -c -k --passes 300 -b 24 --ect 16 -f vw/train.model.vw
vw vw/cv.vw -t -i vw/train.model.vw -p cv.preds.txt
"""


import os
import sys
import string
import random

import numpy as np

from nltk.corpus import stopwords

try:
    import pandas as pd
    from pandas import read_csv, DataFrame
except ImportError:
    print 'Pandas library required:'
    print '    sudo pip install pandas'
    exit(1)


random.seed(0)
np.random.seed(0)


def load_training_data(path='train_data'):
    def fname(name):
        return os.path.join(path, name)

    # load data frame from the CSV file
    train = read_csv(fname('train.csv'))
    cv = read_csv(fname('cv.csv'))
    test = read_csv(fname('test.csv'))

    return train, cv, test


def load_test_data(path='data'):
    def fname(name):
        return os.path.join(path, name)

    return read_csv(fname('test.csv'))


def df_to_vw(df, fname, unlabeled=False):
    columns = [
        #'Id',
        #'ProductId',
        #'UserId',
        #'ProfileName',
        #'HelpfulnessNumerator',
        #'HelpfulnessDenominator',
        #'Time',
        'Summary',
        'Text'
    ]

    stop = stopwords.words('english') + list(string.punctuation) + ['br', 'the', 'this', 'it', 'my']
    stop = map(lambda x: x.decode('utf-8'), stop)

    important = ['like', 'good', 'great', 'taste', 'love', 'best', 'dont', 'bad', 'never']

    def sanitize(text):
        try:
            text =  text.replace('"', '').replace('.', ' . ').replace(',', ' , ').replace("'", '')\
                        .replace(':', 'COLON').replace('|', 'PIPE').replace('  ', ' ').lower()
            for word in important:
                text = text.replace(word, word + ':5.0')
            return ' '.join(t for t in text.split(' ') if t not in stop)
        except AttributeError:
            return str(text)

    if not unlabeled:
        num_per_class = [df['Prediction'].value_counts()[i+1] for i in range(5)]
        total = sum(num_per_class)
        weight_per_class = map(lambda x: total / x, map(float, num_per_class))

    with open(fname, 'w') as fp:
        for row in df.itertuples():
            data = ' |'.join('%s %s' % (i, sanitize(row.__getattribute__(i))) for i in columns)
            if unlabeled:
                fp.write('|%s\n' % data)
            else:
                label = row.Prediction
                weight = weight_per_class[label-1]
                fp.write('%s %s |%s\n' % (label, weight, data))

    if unlabeled:
        with open(fname + '.pred', 'w') as fp:
            preds = pd.DataFrame(df['Prediction'])
            preds.columns = ['Prediction']
            preds.to_csv(fp)


if __name__ != '__main__':
    pred = load_test_data()

    def fname(name, path=None):
        return os.path.join(path or sys.argv[1], '%s.vw' % name)

    df_to_vw(pred, fname('pred'), unlabeled=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2

    train, cv, test = load_training_data()

    def fname(name, path=None):
        return os.path.join(path or sys.argv[1], '%s.vw' % name)

    df_to_vw(train, fname('train'))
    df_to_vw(cv, fname('cv'), unlabeled=True)
    df_to_vw(test, fname('test'), unlabeled=True)
