
###################################
#             WORD2VEC            #
###################################


import pandas as pd
import numpy as np

import os
import sys
import json
import random

from datetime import datetime

np.random.seed(4321)
random.seed(4321)

import data_init
from data_init import batch_generator

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, Merge
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing import sequence


def model1(maxlen, batch_size, num_epochs, w2v, traindf, cvdf):
    train_gen = batch_generator(df=traindf,
                                encoder=lambda b: data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen),
                                batch_size=batch_size,
                                force_batch_size=True)
    cv_gen = batch_generator(df=cvdf,
                             encoder=lambda b: data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen),
                             batch_size=batch_size)

    # creates the neural network
    model = Sequential()
    model.add(LSTM(60, input_dim=300, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(60))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compiles the model
    model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy', 'mse'])

    nb_val_samples = len(cvdf)
    return model, train_gen, cv_gen, nb_val_samples


def model2(maxlen, batch_size, num_epochs, w2v, traindf, cvdf):
    def encoder(b):
        encoded_x, encoded_y = data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen)
        return [[encoded_x, encoded_x, encoded_x], encoded_y]

    train_gen = batch_generator(df=traindf,
                                encoder=encoder,
                                batch_size=batch_size,
                                force_batch_size=True)
    cv_gen = batch_generator(df=cvdf,
                             encoder=encoder,
                             batch_size=batch_size)

    # creates the neural network consisting in 3 convolutional layers going to an LSTM layer
    model1 = Sequential()
    model1.add(Convolution1D(50, 6, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model1.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model1.add(Dropout(0.5))

    model2 = Sequential()
    model2.add(Convolution1D(50, 5, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model2.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model2.add(Dropout(0.5))

    model3 = Sequential()
    model3.add(Convolution1D(50, 4, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model3.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model3.add(Dropout(0.5))

    model = Sequential()
    model.add(Merge([model1, model2, model3], mode='concat', concat_axis=1))
    model.add(LSTM(60, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(60))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compiles the model
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy', 'mse'])

    nb_val_samples = len(cvdf)
    return model, train_gen, cv_gen, nb_val_samples


def model3(maxlen, batch_size, num_epochs, w2v, traindf, cvdf):
    def encoder(b):
        encoded_x, encoded_y = data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen)
        return [[encoded_x, encoded_x, encoded_x], encoded_y]

    train_gen = batch_generator(df=traindf,
                                encoder=encoder,
                                batch_size=batch_size,
                                force_batch_size=True)
    cv_gen = batch_generator(df=cvdf,
                             encoder=encoder,
                             batch_size=batch_size)

    # creates the neural network consisting in 3 convolutional layers going to an LSTM layer
    model1 = Sequential()
    model1.add(Convolution1D(310, 6, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model1.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model1.add(Dropout(0.5))
    model1.add(Convolution1D(100, 3, border_mode='valid', activation='relu'))
    model1.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model1.add(Dropout(0.5))
    model1.add(LSTM(60, return_sequences=True))
    model1.add(Dropout(0.5))

    model2 = Sequential()
    model2.add(Convolution1D(310, 5, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model2.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model2.add(Dropout(0.5))
    model2.add(Convolution1D(100, 3, border_mode='valid', activation='relu'))
    model2.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model2.add(Dropout(0.5))
    model2.add(LSTM(60, return_sequences=True))
    model2.add(Dropout(0.5))

    model3 = Sequential()
    model3.add(Convolution1D(310, 4, border_mode='valid', activation='relu', input_shape=(maxlen, 300)))
    model3.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model3.add(Dropout(0.5))
    model3.add(Convolution1D(100, 3, border_mode='valid', activation='relu'))
    model3.add(MaxPooling1D(pool_length=3, border_mode='valid'))
    model3.add(Dropout(0.5))
    model3.add(LSTM(60, return_sequences=True))
    model3.add(Dropout(0.5))

    model = Sequential()
    model.add(Merge([model1, model2, model3], mode='sum', concat_axis=1))
    model.add(LSTM(60))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # compiles the model
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy', 'mse'])

    nb_val_samples = len(cvdf)
    return model, train_gen, cv_gen, nb_val_samples


def train(model, nn_name, samples_per_epoch, num_epochs, train_gen, cv_gen, nb_val_samples):
    # used to save the model when there was an improvement in the validation set's MSE
    date = datetime.strftime(datetime.today(), '%Y-%m-%d.%H:%M:%S')
    filepath = 'models/%(nn_name)s.%(date)s.h5' % locals()
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='val_mean_squared_error',
                                   verbose=1,
                                   save_best_only=True)
    # TensorBoard logs
    directory = 'logs/%(nn_name)s.%(date)s' % locals()
    if not os.path.exists(directory):
        os.makedirs(directory)
    tensorboard = TensorBoard(log_dir=directory, histogram_freq=0, write_graph=True, write_images=False)

    # trains the model for 1 epoch
    history = model.fit_generator(train_gen,
                                  samples_per_epoch=samples_per_epoch,
                                  nb_epoch=num_epochs,
                                  validation_data=cv_gen,
                                  nb_val_samples=nb_val_samples,
                                  callbacks=[checkpointer, tensorboard])
    return history

if __name__ == '__main__':
    if '--w2v-1' in sys.argv:
        # loads the train and Cross-Validation DataFrames
        print('Loading data...')
        traindf = pd.read_csv('data/train.csv')
        cvdf = pd.read_csv('data/cv.csv')

        print(len(traindf), 'train sequences')
        print(len(cvdf), 'cv sequences')

        maxlen = 400  # all texts are set to this length (either padding or truncating them)
        batch_size = 32  # training batch size
        nn_name = 'w2v-lstmX2-regression'  # name of the NN (used for saving the model and logs)
        num_epochs = 100  # number of epochs to train
        samples_per_epoch = len(traindf) / 50  # texts used in each epoch

        # loads the word2vec model
        print('Loading w2v...')
        w2v = data_init.get_w2v('pretrained/GoogleNews-vectors-negative300.bin')

        # generates the model
        print('Getting model 1...')
        model, train_gen, cv_gen, nb_val_samples = model1(maxlen=maxlen,
                                                          batch_size=batch_size,
                                                          num_epochs=num_epochs,
                                                          w2v=w2v,
                                                          traindf=traindf,
                                                          cvdf=cvdf)

        # trains the model
        print('Training model...')
        train(model, nn_name, samples_per_epoch, num_epochs, train_gen, cv_gen, nb_val_samples)

        # generates the output file
        print('Getting predictions...')
        testdf = pd.read_csv('../data/test.csv')
        testdf = data_init.clean_df(testdf, labeled=False)
        test_encoder = lambda b: data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen, labeled=False)
        data_init.output_results(model, testdf, test_encoder, batch_size)

    elif '--w2v-2' in sys.argv:
        # loads the train and Cross-Validation DataFrames
        print('Loading data...')
        traindf = pd.read_csv('data/train.csv')
        cvdf = pd.read_csv('data/cv.csv')

        print(len(traindf), 'train sequences')
        print(len(cvdf), 'cv sequences')

        maxlen = 400  # all texts are set to this length (either padding or truncating them)
        batch_size = 50  # training batch size
        nn_name = 'w2v-convX3-lstmX2-regression'  # name of the NN (used for saving the model and logs)
        num_epochs = 70  # number of epochs to train
        samples_per_epoch = len(traindf) / 35  # texts used in each epoch

        # loads the word2vec model
        print('Loading w2v...')
        w2v = data_init.get_w2v('pretrained/GoogleNews-vectors-negative300.bin')

        # generates the model
        print('Getting model 2...')
        model, train_gen, cv_gen, nb_val_samples = model2(maxlen=maxlen,
                                                          batch_size=batch_size,
                                                          num_epochs=num_epochs,
                                                          w2v=w2v,
                                                          traindf=traindf,
                                                          cvdf=cvdf)

        # trains the model
        print('Training model...')
        train(model, nn_name, samples_per_epoch, num_epochs, train_gen, cv_gen, nb_val_samples)

        # generates the output file
        print('Getting predictions...')
        testdf = pd.read_csv('../data/test.csv')
        testdf = data_init.clean_df(testdf, labeled=False)
        def test_encoder(b):
            encoded = data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen, labeled=False)
            return encoded, encoded, encoded

        data_init.output_results(model, testdf, test_encoder, batch_size)

    elif '--w2v-3' in sys.argv:
        # loads the train and Cross-Validation DataFrames
        print('Loading data...')
        traindf = pd.read_csv('data/train.csv')
        cvdf = pd.read_csv('data/cv.csv')

        print(len(traindf), 'train sequences')
        print(len(cvdf), 'cv sequences')

        maxlen = 400  # all texts are set to this length (either padding or truncating them)
        batch_size = 50  # training batch size
        nn_name = 'w2v-conv2-lstm-X3-regression'  # name of the NN (used for saving the model and logs)
        num_epochs = 70  # number of epochs to train
        samples_per_epoch = len(traindf) / 35  # texts used in each epoch

        # loads the word2vec model
        print('Loading w2v...')
        w2v = data_init.get_w2v('pretrained/GoogleNews-vectors-negative300.bin')

        # generates the model
        print('Getting model 3...')
        model, train_gen, cv_gen, nb_val_samples = model3(maxlen=maxlen,
                                                          batch_size=batch_size,
                                                          num_epochs=num_epochs,
                                                          w2v=w2v,
                                                          traindf=traindf,
                                                          cvdf=cvdf)

        # trains the model
        print('Training model...')
        train(model, nn_name, samples_per_epoch, num_epochs, train_gen, cv_gen, nb_val_samples)

        # generates the output file
        print('Getting predictions...')
        testdf = pd.read_csv('../data/test.csv')
        testdf = data_init.clean_df(testdf, labeled=False)
        def test_encoder(b):
            encoded = data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen, labeled=False)
            return [encoded, encoded, encoded]

        data_init.output_results(model, testdf, test_encoder, batch_size)
