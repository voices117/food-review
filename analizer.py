# coding: utf-8

import string

import pandas as pd

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess(text):
    return word_tokenize(text.replace("'", '').decode('utf-8'))


if __name__ == '__main__':
    df = pd.read_csv('train_data/train.csv')

    stop = stopwords.words('english') + list(string.punctuation) + ['br', 'the', 'this', 'it', 'my']
    stop = map(lambda x: x.decode('utf-8'), stop)

    count_all = Counter()
    for line in (df.loc[df['Prediction'] == 1])['Text']:
        # Create a list with all the terms
        terms_all = [term for term in preprocess(line.lower()) if term not in stop]
        # Update the counter
        count_all.update(terms_all)
    # Print the first 5 most frequent words
    print(count_all.most_common(40))
