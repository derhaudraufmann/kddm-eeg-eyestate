# here i just do some plotting for presentation purposes

import numpy as np
import matplotlib.pyplot as plt
import arff
from src.util import extract_column


def plot_knn_example():
    data = np.array([1, 1.5, 3, 5, 10, 4, 2, 1, 1, 4, 6,10, 12, 13, 12, 10, 8, 5, 6, 10, 4, 3, 2, 1, 3, 8, 12, 6, 5, 3, 9, 13, 6, 3, 2])

    plt.plot(data)

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("KNN")
    plt.draw()

    plt.show()


def plot_data():
    # read input data from file
    rawData = arff.load(open('../Data/EEGEyeState.arff', 'r'))

    # create numpy array from input data
    data = np.array(rawData['data'])

    col1Data, col1Name = extract_column(rawData, 1)
    col2Data, col2Name = extract_column(rawData, 3)

    plt.plot(data[:, 14])

    plt.plot(col1Data)
    plt.show()