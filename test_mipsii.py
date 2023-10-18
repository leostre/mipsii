import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.datasets import load_wine

import pytest
from time import time

@pytest.fixture(scope='module')
def data():
    return load_wine(return_X_y=True, as_frame=True)

@pytest.fixture(scope='module')
def ys(data):
    _, y = data
    ypred = pd.read_csv('predictions.txt', index_col=0)
    yte = y[ypred.index]
    return yte, ypred

def test_prediction_quality(data, ys):
    # X, y = data #load_wine(return_X_y=True, as_frame=True)
    # ypred = pd.read_csv('predictions.txt', index_col=0)
    # yte = y[preds.index]
    yte, ypred = ys

    score = f1_score(yte, ypred, average='weighted')
    print('F1 score:', score)
    assert score > .85, 'The model quality is too low!'

def test_precision_recall(ys):
    # X, y = data #load_wine(return_X_y=True, as_frame=True)
    # ypred = pd.read_csv('predictions.txt', index_col=0)
    # yte = y[preds.index]
    yte, ypred = ys

    pr = precision_score(yte, ypred, average='weighted')
    re = recall_score(yte, ypred, average='weighted')
    print('Precision score:', pr)
    print('Recall score:', re)
    assert re > .8, 'The model recall is too low!'
    assert pr > .8, 'The model precision is too low!'
    

