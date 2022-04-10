# source: https://www.youtube.com/watch?v=3eZ56HERVbk
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN




df = pd.read_csv('HistoricalPrices.csv')
df = df[' Close']


# reverse dataframe
df = df.loc[::-1]
print(df.head())

# get change between days
l = df.tolist()

diffs = []
for i in range(1,len(l)):
    diffs.append((l[i] - l[i-1])/l[i-1])

df2 = pd.DataFrame (diffs, columns = ['diffs'])
# split data into training and test
values = df2.values

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
model.compile(loss='huber_loss', metrics=['mean_absolute_error'], optimizer='adam')
model.summary()

# create these folders if they does not exist for logging
if not os.path.isdir("results"):
    os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")


# train the model
model.fit(
    trainX,
    trainY,
    epochs=50,
    batch_size=64,
    validation_data=(testX, testY),
    verbose=2
)

# predict the price and plot
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
predicted=np.concatenate((trainPredict, testPredict), axis=0)

days = list(range(len(predicted)))
vals1 = list(values[step:6000])
vals2 = list(values[6000+step::])

# use vals3 to compare to predicted values
vals3 = vals1 + vals2

# check the accuracy
correct = 0
incorrect = 0
for i in range(len(predicted)):
    if predicted[i] > 0 and vals3[i] > 0:
        correct += 1
    elif predicted[i] < 0 and vals3[i] < 0:
        correct += 1

    else:
        incorrect += 1

accuracy = correct/(len(predicted))
print("Correct: ", correct)
print("Incorrect: ", incorrect)
print("Total Prediction Accuracy: ", accuracy)

plt.plot(days, vals3, label='sp500')
plt.plot(days, predicted, label='predicted')
plt.legend(loc="upper left")
plt.show()
