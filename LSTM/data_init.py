"""
Converts the text data into a sequence of 1 hot encoded vectors for each
"""

import pdb
import re
import sys
import string
import random

import numpy as np
import pandas as pd

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.utils import np_utils
from keras.preprocessing import sequence


HTML_RE = re.compile(r'<.{1,10}>')
URL_RE = re.compile(r'(?:https?:)//www\.\w+\.\w+[\w/]+')
MONEY_RE = re.compile(r'\$+(?:[0-9]+)?')
DATE_RE = re.compile(r'\d{1,2}/\d+/\d+')
TIME_RE = re.compile(r'\d\d\:\d\d(?:\:\d\d)?')
FREQ_RE = re.compile(r'([0-9]{1,2})([a-z]{2,})')
NUMBER_RE = re.compile(r'[0-9]+(?:\.|,)?[0-9]+')
REMOVE_RE = re.compile(r'[^\x13-\x7f]|[-"~=_\\#*+@^`|:{}\[\]<>;/]')
COMMA_RE = re.compile(r'(,|\.)')
GROUP_REPLACE_RE = re.compile(r"(\'[a-z]{1,2})")
QUOTED_TEXT_RE = re.compile(r' ("|\')(.*?)\1 ')
QUOTES_REMOVE_RE = re.compile(r"'\w{3,}")
SIGN_RE = re.compile(r'[()\!?]')
POSSESIVE_RE = re.compile(r"([a-z]{3,})'s")
SPACE_RE = re.compile(r' {2,}')


def clean(text):
    text = text.decode('utf-8').lower()

    text = HTML_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = DATE_RE.sub('date', text)
    text = TIME_RE.sub('time', text)
    text = GROUP_REPLACE_RE.sub(lambda m: m.group(1), text)

    text = MONEY_RE.sub('money', text)
    text = FREQ_RE.sub(lambda m: '%s %s' % (m.group(1), m.group(2)), text)

    # numbers are replaced by the word "number"
    text = NUMBER_RE.sub('number', text)

    # removes non ASCII characters and some patterns like ----, ~~~~, ____ etc
    text = REMOVE_RE.sub(' ', text)

    # adds a space after commas that are next to a character
    text = COMMA_RE.sub(lambda m: ' %s ' % m.group(1), text)

    text = SIGN_RE.sub(lambda m: ' %s ' % m.group(0), text)

    # removes quotes around text
    text = QUOTED_TEXT_RE.sub(lambda m: ' ' + m.group(2) + ' ', text)
    text = QUOTES_REMOVE_RE.sub(lambda m: m.group(0).replace("'", ' '), text)

    text = POSSESIVE_RE.sub(lambda m: m.group(1), text)

    return SPACE_RE.sub(' ', text)


def tokenize(text):
    # tokenizes the result
    return word_tokenize(clean(text))


def chunker(iterable, chunk_size):
    """Helper function to iterate a sequence by chunks of `chunk_size`.

    >>> list(chunker('chunk this string', 4))
    ['chun', 'k th', 'is s', 'trin', 'g']
    """
    result = []
    for item in iterable:
        result.append(item)
        if len(result) == chunk_size:
            yield np.array(result)
            result = []
    if len(result) > 0:
        yield np.array(result)


def batch_generator(df, encoder, batch_size, force_batch_size=False, shuffle=True):
    """Creates a generator for batches of `batch_size` encoding the input data frame `df` using
    the `encoder` callable.
    `encoder` receives a DataFrame with columns 'Text' and 'Prediction' and should return a tuple
    with the encoded values for the text and prediction to be fed into the neural network.
    if `force_batch_size` is True, then the last batch will be filled with the first elements of
    `df` if necessary to be of `batch_size`.
    WARNING: this generates a never ending sequence of batches repeating `df`."""

    while True:
        idx = list(df.index)
        if shuffle:
            random.shuffle(idx)

        if len(idx) % batch_size and force_batch_size:
            missing_elems = batch_size - (len(idx) % batch_size)
            idx += idx[:missing_elems]

        for batch_idx in chunker(idx, batch_size):
            batch = df.ix[batch_idx]
            yield encoder(batch)


def balanced_batch_generator(df, encoder, batch_size, force_batch_size=False, shuffle=True):
    """Similar to `batch_generator` but  balances the epoch batches so the 5 start reviews are not
    so dominant."""

    while True:
        rating5 = df[df['Prediction'] == 5]
        idx5 = list(rating5.index)
        random.shuffle(idx5)

        for r5batch in chunker(idx5, len(idx5) / 4):
            non5 = df[df['Prediction'] != 5]

            idx = list(non5.index) + list(r5batch)
            if shuffle:
                random.shuffle(idx)

            if len(idx) % batch_size and force_batch_size:
                missing_elems = batch_size - (len(idx) % batch_size)
                idx += idx[:missing_elems]

            for batch_idx in chunker(idx, batch_size):
                batch = df.ix[batch_idx]
                yield encoder(batch)


def one_hot(text, alphabet):
    """Does a one hot encoding of the text.
    The values of each character go from 1 to len(alphabet)"""

    # create mapping of characters to integers and the reverse
    char_to_int = dict((c, i+1) for i, c in enumerate(alphabet))
    int_to_char = dict((i+1, c) for i, c in enumerate(alphabet))

    return map(lambda e: char_to_int[e], text)


def get_alphabet(df, alphabet):
    alphabet = set()
    for text in df['Text']:
        alphabet |= set(text)
    return alphabet


def encode_labeled_data(df, alphabet=None, maxlen=450):
    """Returns a tuple with 2 elements:
    an array containing the encoded texts.
    a list containing the predictions for each text."""

    train_data = []
    train_labels = []
    alphabet = alphabet or get_alphabet(df)
    for text, pred in zip(df.Text, df.Prediction):
        # does the one hot encoding of the text
        text = one_hot(text, alphabet)

        if len(text) > maxlen:
            # clips the text
            text = text[:maxlen]
        elif len(text) < maxlen:
            # pads the text with zeros
            text = [0] * (maxlen - len(text)) + text

        train_data.append(text)

        # subtract 1 to make the prediction range start from zero instead of 1
        train_labels.append(pred - 1)

    # reshapes to the LSTM expected input format
    train_data = np.reshape(train_data, (len(train_data), maxlen, 1)).astype('float32')

    # normalize
    train_data /= float(len(alphabet))

    return train_data, np_utils.to_categorical(train_labels)


def encode_labeled_data_regression(df, alphabet=None, maxlen=450):
    train_data = []
    train_labels = []
    alphabet = alphabet or get_alphabet(df)
    for text, pred in zip(df.Text, df.Prediction):
        # does the one hot encoding of the text
        text = one_hot(text, alphabet)

        if len(text) > maxlen:
            # clips the text
            text = text[:maxlen]
        elif len(text) < maxlen:
            # pads the text with zeros
            text = [0] * (maxlen - len(text)) + text

        train_data.append(text)

        # normalize the prediction to a 0-1 range
        train_labels.append(float(pred - 1) / 4)

    # reshapes to the LSTM expected input format
    train_data = np.reshape(train_data, (len(train_data), maxlen, 1)).astype('float32')

    # normalize
    train_data /= float(len(alphabet))

    train_labels = np.asarray(train_labels)
    return train_data, train_labels.reshape((len(train_labels), 1))


def encode_labeled_data_words(df, alphabet=None, maxlen=450):
    wordc = [1]
    words = alphabet or {}
    train_data, train_labels = [], []
    max_len_found = 0

    def word_to_int(w):
        if w not in words:
            # if the alphabet was send by the user, it's not modified
            if alphabet is not None:
                # zero is ignored
                return 0
            words[w] = wordc[0]
            wordc[0] += 1
        return words[w]

    c = 1
    for text, pred in zip(df.Text, df.Prediction):
        # cleans and tokenizes the text
        text = tokenize(text)
        max_len_found = max(len(text), max_len_found)

        # maps each word to an integer
        encoded = []

        train_data.append(map(word_to_int, text))
        train_labels.append(float(pred - 1) / 4)
        print 'encoding %s/%s\r' % (c, len(df)),
        c += 1

    print
    if alphabet is None:
        print 'found %s words' % len(words)
        print 'longest review had %s words' % max_len_found

    train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
    return train_data, np.array(train_labels), words


def encode_unlabeled_data_words(df, alphabet, maxlen=400):
    wordc = [1]
    words = alphabet
    train_data = []

    def word_to_int(w):
        if w not in words:
            # if the alphabet was send by the user, it's not modified
            if alphabet is not None:
                # zero is ignored
                return 0
            words[w] = wordc[0]
            wordc[0] += 1
        return words[w]

    c = 1
    for text in df.Text:
        # cleans and tokenizes the text
        text = tokenize(text)

        # maps each word to an integer
        encoded = []

        train_data.append(map(word_to_int, text))
        print 'encoding %s/%s\r' % (c, len(df)),
        c += 1

    print
    print 'found %s examples' % len(train_data)

    train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
    return train_data


def get_embed_alphabet(df):
    alphabet = {}
    stop = stopwords.words('english')
    c = [1]

    def get_alph(text):
        tokens = word_tokenize(text)
        for token in tokens:
            if token not in alphabet:
                alphabet[token] = c[0]
                c[0] += 1

    df['Text'].apply(get_alph)
    return alphabet


def encode_embed(df, maxlen, alphabet=None, labeled=True, categorical=False):
    words = alphabet or {}
    stop = stopwords.words('english')

    def text2vec(text):
        output = []
        tokens = word_tokenize(text)

        for token in tokens:
            #if token in stop:
            #    continue

            encoded = alphabet.get(token, 0)
            output.append(encoded)

        # clips or pads the output to the max length
        if len(output) < maxlen:
            output = [0] * (maxlen - len(output)) + output
        elif len(output) > maxlen:
            output = output[:maxlen]

        return np.asarray(output, dtype='float32').reshape(1, maxlen)

    X = df['Text'].apply(text2vec)
    X = np.asarray(list(X)).reshape((len(df), maxlen))

    if labeled:
        if categorical:
            preds = np.asarray(df['Prediction'] - 1)
            y = np_utils.to_categorical(preds, 5)
        else:
            y = (df['Prediction'] - 1) / 4
            y = np.asarray(y)
        return X, y
    else:
        return X


def get_w2v(filename):
    """Gets the decoded word2vec dictionary from `filename`.
    Returns a dict containing for each word, the corresponding numpy ndarray."""

    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    return word_vecs


def encode_w2v(df, w2v, maxlen=450, labeled=True, categorical=False):
    """Converts a dataframe encoded by `clean_df` into 300x1 word vecs from Google (Mikolov)
    word2vec.
    `df` is The DataFrame to encode, `w2v` is the word2vec binary file name.
    The texts in `df` are truncated or padded to `maxlen`.
    `categorical` formats the labels for a classification problem instead of regression."""

    if isinstance(w2v, basestring):
        word_vecs = get_w2v(w2v)
    else:
        word_vecs = w2v

    word_vec_len = w2v.values()[0].shape[0]
    assert word_vec_len == 300, w2v.values()[0].shape

    def text2vec(text):
        output = []
        tokens = word_tokenize(text)

        for token in tokens:
            if token in word_vecs:
                output.append(word_vecs[token])
            if len(output) == maxlen:
                break

        # completes the missing characters until maxlen is reached
        padding = []
        for i in range(maxlen - len(output)):
            padding.append(np.asarray([0] * word_vec_len))

        output = padding + output
        return np.asarray(output, dtype='float32').reshape(maxlen, word_vec_len)

    X = df['Text'].apply(text2vec)
    X = np.asarray(list(X)).reshape((len(df), maxlen, word_vec_len))

    if labeled:
        if categorical:
            preds = np.asarray(df['Prediction'] - 1)
            y = np_utils.to_categorical(preds, 5)
        else:
            y = (df['Prediction'] - 1) / 4
            y = np.asarray(y)
        return X, y
    else:
        return X


def clean_df(df, labeled=True):
    """Returns a new DataFrame where the `Summary` and `Text` columns are merged and the text is
    cleaned (see `clean`).
    Use `labeled` as False if the DataFrame is meant for predictions."""

    # adds the summary and the text as a unique field separated by a space
    # some reviews have a "N/A" summary, so it's just set to an empty string
    out = df['Summary'].fillna(' ') + ' ' + df['Text']

    # cleans and tokenizes the `Summary` + `Text` in each row
    out = out.apply(clean)

    if labeled:
        # joins the tokenized text with the corresponding prediction
        out = pd.concat([out, df['Prediction']], axis=1)
        out.columns = ['Text', 'Prediction']
    else:
        out = pd.concat([df['Id'], out], axis=1)
        out.columns = ['Id', 'Text']
    return out


def output_results(model, testdf, encoder, batch_size):
    """Helper function that given the test set, predicts the values and generates the CSV file ready
    to be uploaded to Kaggle.
    `model` is the trained model used to predict the results.
    `testdf` is the DataFrame to generate the predictions.
    `encoder` is the batch encoding function used by the batch generator."""

    assert 'Text' in testdf.columns
    assert 'Id' in testdf.columns

    print 'Got %s texts' % len(testdf)

    test_gen = batch_generator(df=testdf, encoder=encoder, batch_size=batch_size, shuffle=False)

    print 'Getting predictions...'
    preds = model.predict_generator(test_gen, val_samples=len(testdf))

    print 'Got %s predictions' % len(preds)

    print 'Generating output...'
    # converts the unnormalized predictions to a DataFrame
    df = pd.DataFrame(preds * 4 + 1)
    df.columns = ['Prediction']

    # saves as Kaggle's format
    res = pd.concat([testdf['Id'], df['Prediction']], axis=1)
    res.to_csv('results.csv', index=False)


if __name__ == '__main__':
    if '--clean' in sys.argv:
        files = {
            '../train_data/train.csv': 'data/train.csv',
            '../train_data/cv.csv': 'data/cv.csv',
            '../train_data/test.csv': 'data/test.csv',
            '../data/test.csv': 'data/kaggle_test.csv',
        }

        # encodes the datasets
        for src, dst in files.items():
            print '%s -> %s' % (src, dst)
            df = pd.read_csv(src)
            df = clean_df(df)
            df.to_csv(dst)
    elif '--w2v' in sys.argv:
        files = {
            'data/train.csv': 'data/train.w2v.csv',
            'data/cv.csv': 'data/cv.w2v.csv',
            'data/test.csv': 'data/test.w2v.csv',
        }

        # encodes the datasets
        for src, dst in files.items():
            print '%s -> %s' % (src, dst)
            df = pd.read_csv(src)
            df = clean_df(df)
            df.to_csv(dst)
    elif '--unittest' in sys.argv:
        test = """Axel's:1234 another 1324.4 (review),  [bye bye brackets] 6pack 11/08/1990 text goes 11:23:54 ,in .here "this shouldn't be 1234 quoted!!!" $$ 'nor this text's words'. <br /> <p>HTML gone</p> please... remove this url http://www.google.com $1000"""
        expected = """axel number another number ( review ) , bye bye brackets 6 pack date text goes time , in . here this shouldn't be number quoted ! ! ! money nor this text words . html gone please . . . remove this url money"""
        obtained = clean(test)
        assert expected == obtained, obtained
        print 'OK!'
    else:
        assert 0, 'No input arg found'
