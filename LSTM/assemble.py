import numpy as np
import pandas as pd

from keras.models import load_model

import data_init


def assemble_rc(cls_preds, reg_preds, threshold):
    """Returns new predictions by combining a regression (`reg_preds`) and classification
    (`cls_preds`) predictions.
    Both inputs should be numpy arrays.
    The predictions are returned as a DataFrame."""

    assert cls_preds.shape == reg_preds.shape, '%s != %s' % (cls_preds.shape, reg_preds.shape)

    clsdf = pd.DataFrame(cls_preds)
    regdf = pd.DataFrame(reg_preds)

    clsdf.columns = ['Prediction']
    regdf.columns = ['Prediction']

    # gets the difference between predictions
    delta = np.abs(cls_preds - reg_preds)
    deltadf = pd.DataFrame(delta)
    deltadf.columns = ['Prediction']

    # if the difference is under the threshold, then the classification result is used
    # for example, 4.89 and 5 -> 0.11 < 0.3 (threshold) -> 5 is the final prediction
    reg_res = regdf[(deltadf['Prediction'] != 5) | (deltadf['Prediction'] > threshold)]
    cls_res = clsdf[(deltadf['Prediction'] == 5) & (deltadf['Prediction'] <= threshold)]

    return pd.concat([reg_res, cls_res])


def get_model_preds(filename, testdf, maxlen, batch_size, encoder):
    model = load_model(filename)

    print 'Getting regression predictions for %s...' % filename
    test_gen = data_init.batch_generator(df=testdf, encoder=encoder, batch_size=batch_size, shuffle=False)

    return model.predict_generator(test_gen, val_samples=len(testdf))


if __name__ == '__main__':
    testdf = pd.read_csv('../data/test.csv')
    testdf = data_init.clean_df(testdf, labeled=False)

    maxlen, batch_size = 400, 50

    print('Loading w2v...')
    w2v = data_init.get_w2v('pretrained/GoogleNews-vectors-negative300.bin')

    def encoder(b):
        encoded = data_init.encode_w2v(df=b, w2v=w2v, maxlen=maxlen, labeled=False)
        return [encoded, encoded, encoded]

    # gets predictions from the different models
    reg_preds = get_model_preds(filename='best/w2v-conv2-lstm-X3-regression.2016-11-14.08:05:10.h5',
                                testdf=testdf,
                                maxlen=maxlen,
                                batch_size=batch_size,
                                encoder=encoder)
    cls_preds = get_model_preds(filename='best/w2v-conv2-lstm-X3-categorical.2016-11-16.07:52:01.h5',
                                testdf=testdf,
                                maxlen=maxlen,
                                batch_size=batch_size,
                                encoder=encoder)

    # sets predictions to the normal 1-5 range
    reg_preds = (reg_preds * 4) + 1
    cls_preds = (np.argmax(cls_preds, axis=1) + 1).reshape(reg_preds.shape)

    print 'Assembling results...'
    results = assemble_rc(cls_preds=cls_preds, reg_preds=reg_preds, threshold=0.1)

    # sets the original indexes
    results = pd.concat([testdf, results], axis=1)
    assert len(results) == len(testdf), len(results)

    cols = ['Id', 'Prediction']
    results.to_csv('assembled.csv', index=False, columns=cols)

    print 'Done!'
