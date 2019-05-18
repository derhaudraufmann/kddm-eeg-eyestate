import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



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

column = 13
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
X, y = data[:, :-1], data[:, -1]

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)

# evaluate

historyX, historyy = [x for x in trainX], [x for x in trainy]
predictions = list()
for i in range(len(testy)):
	# define model
	model = KNeighborsClassifier(n_neighbors=3)
	# fit model on a small subset of the train set
	tmpX, tmpy = np.array(historyX)[-10:,:], np.array(historyy)[-10:]
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
print(score)