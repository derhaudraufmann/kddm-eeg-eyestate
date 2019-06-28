# classification with Support-Vector-Machine, did not produce satisfying results and thus was abandoned

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier

import scipy
import pywt
from collections import Counter

from src.util import extract_column, load_data


def eval_svm():
    data, rawData = load_data()
    col1Data, col1Name = extract_column(rawData, 2)

    col0Data, col0Name = extract_column(rawData, 0)
    col1Data, col1Name = extract_column(rawData, 1)
    col2Data, col2Name = extract_column(rawData, 2)
    col3Data, col3Name = extract_column(rawData, 3)

    # split dataset
    # X, y = col1Data, data[:, -1]

    X = np.column_stack((col0Data, col1Data, col2Data, col3Data))
    y = data[:, -1]

    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)

    svclassifier = SVC(kernel='rbf', C=2)
    svclassifier.fit(trainX, trainy)

    y_pred = svclassifier.predict(testX)

    print(confusion_matrix(testy, y_pred))
    print(classification_report(testy, y_pred))


def wavelet_trans():
    data, rawData = load_data()

    col0Data, col0Name = extract_column(rawData, 0)
    col1Data, col1Name = extract_column(rawData, 1)
    col2Data, col2Name = extract_column(rawData, 2)
    col3Data, col3Name = extract_column(rawData, 3)
    col4Data, col3Name = extract_column(rawData, 4)
    col5Data, col3Name = extract_column(rawData, 5)
    col6Data, col3Name = extract_column(rawData, 6)
    col7Data, col3Name = extract_column(rawData, 7)
    col8Data, col3Name = extract_column(rawData, 8)
    col9Data, col3Name = extract_column(rawData, 9)
    col10Data, col3Name = extract_column(rawData, 10)
    col11Data, col3Name = extract_column(rawData, 11)
    col12Data, col3Name = extract_column(rawData, 12)
    col13Data, col3Name = extract_column(rawData, 13)

    # split dataset
    # X, y = col1Data, data[:, -1]

    X = np.column_stack(
        (col0Data, col1Data, col2Data, col3Data, col4Data, col5Data, col6Data, col7Data, col8Data, col9Data, col10Data, col11Data, col12Data, col13Data))
    y = data[:, -1]

    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)

    X_window = np.reshape(X[:-4], (1872, 8, X.shape[1]))

    kfold = KFold(n_splits=5, random_state=1, shuffle=False)

    result = np.array([])
    for train_index, test_index in kfold.split(X_window):
        print('TRAIN:' + str(train_index) + 'TEST:' + str(test_index))
        # X_train, X_test = X_window[train_index], X_window[test_index]
        # y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train_ucihar, Y_train_ucihar = get_uci_har_features_windowed(X_train, y_train, 'haar') #rbio3.1
        # X_test_ucihar, Y_test_ucihar = get_uci_har_features_windowed(X_test, y_test, 'haar')

        # X_train_ucihar, Y_train_ucihar = get_uci_har_features(X_train, y_train, 'rbio3.1')
        # X_test_ucihar, Y_test_ucihar = get_uci_har_features(X_test, y_test, 'rbio3.1')

        cls = GradientBoostingClassifier(n_estimators=500, learning_rate=0.125, min_samples_split=1000, min_samples_leaf=1, max_depth=5)
        cls.fit(X_train, y_train)
        train_score = cls.score(X_train, y_train)
        test_score = cls.score(X_test, y_test)
        # cls.fit(X_train_ucihar, Y_train_ucihar)
        # train_score = cls.score(X_train_ucihar, Y_train_ucihar)
        # test_score = cls.score(X_test_ucihar, Y_test_ucihar)
        print("Train Score for the ECG dataset is about: {}".format(train_score))
        print(str(test_score))
        result = np.append(result, test_score)
    print("Overall results")
    print("Mean:" + str(np.mean(result)))
    print("Median:" + str(np.median(result)))
    print("Min:" + str(np.min(result)) + " , max:" + str(np.max(result)))


# eval_svm()
wavelet_trans()
