
# source: https://www.youtube.com/watch?v=3eZ56HERVbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


df = pd.read_csv('HistoricalPrices.csv')
df = df[' Close']
# reverse dataframe
df = df.loc[::-1]
print(df.head())

# split data into training and test
values = df.values

# train, test = values[0:Tp,:], values[Tp:N,:]
train, test = values[0:6000], values[6000::]
step = 5


# convert into dataset matrix
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i +step
        X.append(data[i:d])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test,step)

#reshape data to be 3-D

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape
testX.shape

#build model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1,step), activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.summary()

# predict and plot result
model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=2)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
predicted=np.concatenate((trainPredict, testPredict), axis=0)

# check the loss
trainScore = model.evaluate(trainX, trainY, verbose=0)

days = list(range(len(predicted)))
vals1 = list(values[step:6000])
vals2 = list(values[6000+step::])

# use vals3 to compare to predicted values
vals3 = vals1 + vals2

plt.plot(days, vals3, label='sp500')
plt.plot(days, predicted, label='predicted')
plt.legend(loc="upper left")
plt.show()
