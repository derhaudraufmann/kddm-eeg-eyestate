# trying out a wavelet transformation approach from http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# this produced results combined with KNN and Gradient Boost, but in the end worked worse than using Gradient Boost
# directly with the input signal. instead of it's transformation  thus was abandoned.



#wavelet transformation taken from http://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
# and modified for my purposes

import numpy as np
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier

import scipy
import pywt
from collections import Counter

from src.util import extract_column, load_data

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def get_features(list_values):
    entropy = calculate_entropy(list_values)
    # crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + statistics


def get_transformed_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        signal = dataset[signal_no]
        list_coeff = pywt.wavedec(signal, waveletname)
        for coeff in list_coeff:
            features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y

def get_transformed_features_windowed(dataset, labels, waveletname):
    ret_features = []
    for signal_no in range(0, len(dataset)):
        features = []
        for signal_comp in range(0, dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            list_coeff = pywt.wavedec(signal, waveletname)
            for coeff in list_coeff:
                features += get_features(coeff)
        ret_features.append(features)
    X = np.array(ret_features)
    Y = np.array(labels)
    return X, Y


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

    X = np.column_stack(
        (col0Data, col1Data, col2Data, col3Data, col4Data, col5Data, col6Data, col7Data, col8Data, col9Data, col10Data, col11Data, col12Data, col13Data))
    y = data[:, -1]


    X_window = np.reshape(X[:-4], (1872, 8, X.shape[1]))

    kfold = KFold(n_splits=5, random_state=1, shuffle=False)

    result = np.array([])
    for train_index, test_index in kfold.split(X_window):
        print('TRAIN:' + str(train_index) + 'TEST:' + str(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_ucihar, Y_train_ucihar = get_transformed_features(X_train, y_train, 'rbio3.1')
        X_test_ucihar, Y_test_ucihar = get_transformed_features(X_test, y_test, 'rbio3.1')

        cls = GradientBoostingClassifier(n_estimators=500, learning_rate=0.125, min_samples_split=1000, min_samples_leaf=1, max_depth=5)
        cls.fit(X_train_ucihar, Y_train_ucihar)

        train_score = cls.score(X_train_ucihar, Y_train_ucihar)
        test_score = cls.score(X_test_ucihar, Y_test_ucihar)
        print("Train Score for the ECG dataset is about: {}".format(train_score))
        print(str(test_score))
        result = np.append(result, test_score)

    print("Overall results")
    print("Mean:" + str(np.mean(result)))
    print("Median:" + str(np.median(result)))
    print("Min:" + str(np.min(result)) + " , max:" + str(np.max(result)))


wavelet_trans()