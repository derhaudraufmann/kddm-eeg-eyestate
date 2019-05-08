import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
