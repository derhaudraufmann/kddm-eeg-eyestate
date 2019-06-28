# classification with Gradient boost, this approach in the end resulted in 74% accuracy and was used as final result

import numpy as np
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier

from src.util import extract_column, load_data

def gradient_boot():
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

        cls = GradientBoostingClassifier(n_estimators=500, learning_rate=0.125, min_samples_split=1000, min_samples_leaf=1, max_depth=5)
        cls.fit(X_train, y_train)
        train_score = cls.score(X_train, y_train)
        test_score = cls.score(X_test, y_test)
        print("Train Score for the ECG dataset is about: {}".format(train_score))
        print(str(test_score))
        result = np.append(result, test_score)
    print("Overall results")
    print("Mean:" + str(np.mean(result)))
    print("Median:" + str(np.median(result)))
    print("Min:" + str(np.min(result)) + " , max:" + str(np.max(result)))


gradient_boot()
