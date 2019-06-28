# in here, we are classifying with a simple KNN classifier.
# this resultet in great restuls when shuffling the dataset (97%)
# but really bad results (~50%) when not shuffling

# this is mainly to reproduce the findings of the originial authors(Rösler et al.) of the dataset
# as well as the findings of: https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/
# who showed that this result is invalid due to the test methology


import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.util import extract_column, load_data

data, rawData = load_data()

col1Data, col1Name = extract_column(rawData, 2)
col2Data, col2Name = extract_column(rawData, 3)

# plot autocorrelation
autocorrelation_plot(col1Data)


# split dataset
X, y = col1Data, data[:, -1]
# X = np.column_stack((col1Data,col2Data))
X, y = data[:, :-1], data[:, -1]
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)


# reproduce findings of: https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/

historyX, historyy = [x for x in trainX], [x for x in trainy]
predictions = list()
for i in range(len(testy)):
	# define model
	model = KNeighborsClassifier(n_neighbors=3)
	# fit model on a small subset of the train set
	tmpX, tmpy = np.array(historyX), np.array(historyy)
	model.fit(tmpX, tmpy)
	# forecast the next time step
	yhat = model.predict([testX[i, :]])[0]
	# store prediction
	predictions.append(yhat)
	# add real observation to history
	historyX.append(testX[i, :])
	historyy.append(testy[i])
# evaluate predictions
score = accuracy_score(testy, predictions)
print("KNN Score:")
print(score)




# # KNN with temporal ordering
predictions = list()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(trainX, trainy)

for i in range(len(testy)):
	# define model
	# fit model on a small subset of the train set

	# forecast the next time step
	yhat = model.predict([testX[i, :]])[0]
	# store prediction
	predictions.append(yhat)
# evaluate predictions
score = accuracy_score(testy, predictions)
print("KNN Score:")
print(score)


plt.plot(predictions)
plt.plot(testX[:, :1])
plt.title(col1Name)
dpi = 400
fig1 = plt.gcf()
fig1.set_size_inches(100, 30)
plt.draw()
fig1.savefig('../findings/channel_vs_eyes_col' + str(2) + '.pdf', dpi=dpi)

plt.show()


print("done")