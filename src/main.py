import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam



from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

# outlier detection based on median
# source: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
# took second answer and adpated to outliers are not removed, but set to median
# high variance is ok, since outliers seem to be very extreme
def reject_outliers(data, m = 40.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    data[s>=m] = np.median(data)
    return data

# read input data from file
rawData = arff.load(open('../Data/EEGEyeState.arff', 'r'))

# create numpy array from input data
data = np.array(rawData['data'])

column = 5
colName = rawData['attributes'][column][0]
colData = data[:, column]

# cast to float(from string)
colData = colData.astype(np.float)

# outlier removal
colData = reject_outliers(colData)

# normalization (min-max, 0-1)
colData = colData.reshape(-1,1)
scaler = MinMaxScaler()
scaler.fit(colData)
MinMaxScaler(copy=True, feature_range=(0, 1))
colData = scaler.transform(colData)

plt.plot(data[:, 14])

plt.plot(colData)
plt.title(colName)
dpi = 400
fig1 = plt.gcf()
fig1.set_size_inches(100, 30)
plt.draw()
fig1.savefig('../findings/channel_vs_eyes_col' + str(column) + '.pdf', dpi=dpi)

plt.show()

# autocorrelation
autocorrelation_plot(colData)
pyplot.show()

# KNN classification
# source: https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/
#

# split dataset
X, y = colData, data[:, -1]
# X, y = data[:, :-1], data[:, -1]

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)


# evaluate

historyX, historyy = [x for x in trainX], [x for x in trainy]
# predictions = list()
# for i in range(len(testy)):
# 	# define model
# 	model = KNeighborsClassifier(n_neighbors=3)
# 	# fit model on a small subset of the train set
# 	tmpX, tmpy = np.array(historyX)[-10:,:], np.array(historyy)[-10:]
# 	model.fit(tmpX, tmpy)
# 	# forecast the next time step
# 	yhat = model.predict([testX[i, :]])[0]
# 	# store prediction
# 	predictions.append(yhat)
# 	# add real observation to history
# 	historyX.append(testX[i, :])
# 	historyy.append(testy[i])
# # evaluate predictions
# score = accuracy_score(testy, predictions)
# print("KNN Score:")
# print(score)
#
#
# # MLp
# predictions = list()
# for i in range(len(testy)):
#     # define model
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#     # fit model on a small subset of the train set
#     tmpX, tmpy = np.array(historyX)[-10:, :], np.array(historyy)[-10:]
#     clf.fit(tmpX.astype(np.float64), tmpy)
#     # forecast the next time step
#     yhat = clf.predict([testX[i, :].astype(np.float64)])[0]
#     # store prediction
#     predictions.append(yhat)
#     # add real observation to history
#     historyX.append(testX[i, :])
#     historyy.append(testy[i])
# # evaluate predictions
# score = accuracy_score(testy, predictions)
# print("MLP Score:")
# print(score)

# MLP 2
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# # fit model on a small subset of the train set
# clf.fit(trainX.astype(np.float64), trainy)
#
# predictions = list()
# for i in range(len(testy)):
#     # forecast the next time step
#     yhat = clf.predict([testX[i, :].astype(np.float64)])[0]
#     # store prediction
#     predictions.append(yhat)
# # evaluate predictions
# score = accuracy_score(testy, predictions)
# print("MLP Score:")
# print(score)

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
	plt.title(dataset_name + "channel: " + str(column)  + "accuracy: " + str(accuracy))
	plt.legend()

	# Ensure the plot shows correctly.
	# plt.savefig("../../findings/plots/" + plotFileName() + "_history.png")
	plt.show()

# reshape data to fit LSTM
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#LSTM
def build_model():
	# prepare data
	# define parameters
	verbose, epochs, batch_size = 0, 200, 16
	n_timesteps, n_features, n_outputs = 10, 14, 1
	# define model
	model = Sequential()
	model.add(LSTM(200, return_sequences=False, input_shape=(1, trainX.shape[1])))
	model.add(Dropout(0.5))
	model.add(Dense(n_outputs, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['binary_accuracy'])



	# fit network
	history = model.fit(X_train, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)
	return model, history


lstm, training_history = build_model()


result = lstm.evaluate(X_test, testy, verbose=0)

plot_training_history(training_history, result[1])

print("lstm scores:")
print("loss: " + str(result[0]) + " , acc: " + str(result[1]))

#LSTM2
#
# model = Sequential()
# model.add(Embedding(trainX.shape[0], embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, epochs=3, batch_size=64)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)

print("done")