# in this file i tried applying an artifical neural network approach, namely a simple multi-layer-perceptron
# and a Long-Short-Term-Memory network to get temporal context
# this did not produce usable results and was abandoned

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop

from pandas import DataFrame
from pandas import concat

from src.util import extract_column, load_data
data, rawData = load_data()

col1Data, col1Name = extract_column(rawData, 2)
col2Data, col2Name = extract_column(rawData, 3)

# split dataset
X, y = col1Data, data[:, -1]
# X = np.column_stack((col1Data,col2Data))
X, y = data[:, :-1], data[:, -1]
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)


# MLP 2
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
# fit model on a small subset of the train set
clf.fit(trainX.astype(np.float64), trainy)

predictions = list()
for i in range(len(testy)):
    # forecast the next time step
    yhat = clf.predict([testX[i, :].astype(np.float64)])[0]
    # store prediction
    predictions.append(yhat)
# evaluate predictions
score = accuracy_score(testy, predictions)
print("MLP Score:")
print(score)

def plot_training_history(history, accuracy):
	dataset_name = "EEG Eye Data"
	# Get the classification accuracy and loss-value
	# for the training-set.

	acc = history.history['binary_accuracy']
	loss = history.history['loss']

	# Get it for the validation-set (we only use the test-set).
	val_acc = history.history['val_binary_accuracy']
	val_loss = history.history['val_loss']

	# Plot the accuracy and loss-values for the training-set.
	plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
	plt.plot(loss, 'o', color='b', label='Training Loss')

	# Plot it for the test-set.
	plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
	plt.plot(val_loss, 'o', color='r', label='Test Loss')

	# Plot title and legend.
	plt.title(dataset_name + "channel: " + str(2) + "accuracy: " + str(accuracy))
	plt.legend()

	# Ensure the plot shows correctly.
	# plt.savefig("../../findings/plots/" + plotFileName() + "_history.png")
	plt.show()

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df.values[lag:,:]

# reshape data to fit LSTM

# X_train = timeseries_to_supervised(trainX, 2)
# X_test = timeseries_to_supervised(testX, 2)
# X_train = np.reshape(X_train, (X_train.shape[0], 3, 1))
# X_test = np.reshape(X_test, (X_test.shape[0], 3, 1))

X_train = np.reshape(trainX, (trainX.shape[0], 1, 2))
X_test = np.reshape(testX, (testX.shape[0], 1, 2))

#LSTM
def build_model():
	# prepare data
	# define parameters
	verbose, epochs, batch_size = 0, 600, 16
	n_timesteps, n_features, n_outputs = 10, 14, 1
	# define model
	model = Sequential()
	model.add(LSTM(200, return_sequences=False, input_shape=(1, trainX.shape[1])))
	model.add(Dropout(0.5))
	model.add(Dense(n_outputs, activation='sigmoid'))

	optimizer = RMSprop(lr=1e-4)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])



	# fit network
	history = model.fit(X_train, trainy[:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)
	return model, history


lstm, training_history = build_model()

result = lstm.evaluate(X_test, testy[:], verbose=0)

plot_training_history(training_history, result[1])

print("lstm scores:")
print("loss: " + str(result[0]) + " , acc: " + str(result[1]))

