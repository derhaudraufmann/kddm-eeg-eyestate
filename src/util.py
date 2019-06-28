import numpy as np
from sklearn.preprocessing import MinMaxScaler

import arff


def extract_column(rawData, number):
    data = np.array(rawData['data'])

    colName = rawData['attributes'][number][0]
    col1Data = data[:, number]

    # cast to float(from string)
    col1Data = col1Data.astype(np.float)

    # outlier removal
    col1Data = reject_outliers(col1Data)

    # normalization (min-max, 0-1)
    col1Data = col1Data.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(col1Data)
    MinMaxScaler(copy=True, feature_range=(0, 1))
    col1Data = scaler.transform(col1Data)

    return col1Data, colName


def reject_outliers(data, m = 40.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    data[s>=m] = np.median(data)
    return data

def extract_raw_column(rawData, number):
    data = np.array(rawData['data'])

    colName = rawData['attributes'][number][0]
    col1Data = data[:, number]

    # cast to float(from string)
    col1Data = col1Data.astype(np.float)

    return col1Data, colName

def load_data():
    # read input data from file
    rawData = arff.load(open('../Data/EEGEyeState.arff', 'r'))

    # create numpy array from input data
    return np.array(rawData['data']), rawData

