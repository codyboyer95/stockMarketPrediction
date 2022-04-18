# source: https://www.youtube.com/watch?v=3eZ56HERVbk
# source: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN




df2 = pd.read_csv('HistoricalPrices.csv')

# retain only date and closing price
df2 = df2[['Date', ' Close']]
# convert date to datetime format
df2['Date'] = pd.to_datetime(df2.Date)


df3 = pd.read_csv('new_york_city.csv')

# retain only date and max temp for this test
df3 = df3[['date_time','maxtempC']]
df3.columns = ['Date', 'maxtempC']

# convert date to datetime format
df3['Date'] = pd.to_datetime(df3.Date)

# reverse stock prices dataframe
df2 = df2.loc[::-1]

# merge the dataframes of stock and weather data
df = pd.merge(df2, df3, on="Date", how="right")

# drop rows with NaN
df = df.dropna()

# get changes in stock price into dataframe
df['diffs'] = df[' Close'].sub(df[' Close'].shift()).div(df[' Close'] - 1).fillna(0)
print(df.head())

df = df[['maxtempC', 'diffs']]
values = df.values


# split into train and test sets
values = df.values

n_train = math.floor(len(values)*3/4)
train = values[:n_train, :]
test = values[n_train:, :]


# convert into dataset matrix
step = 5
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i +step
        X.append(data[i:d])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test,step)


# split into input and outputs
#trainX, trainY = train[:, :-1], train[:, -1]
#testX, testY = test[:, :-1], test[:, -1]
trainY = train[5:, -1]
testY = test[5:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = trainX.reshape((trainX.shape[0], 5, 2))
test_X = testX.reshape((testX.shape[0], 5, 2))
print(train_X.shape, trainY.shape, test_X.shape, testY.shape)


#build model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
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
    epochs=200,
    batch_size=64,
    validation_data=(testX, testY),
    verbose=2
)

# predict the price and plot
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# check the accuracy
correct = 0
incorrect = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i in range(len(testPredict)):
    if testY[i] > 0:
        if testPredict[i] > 0:
            correct += 1
            true_pos += 1
        elif testPredict[i] < 0:
            incorrect += 1
            false_neg += 1

    elif testY[i] < 0:
        if testPredict[i] < 0:
            correct += 1
            true_neg += 1
        elif testPredict[i] > 0:
            incorrect += 1
            false_pos += 1


precision = true_pos/(true_pos + false_pos)
recall = true_pos/(true_pos + false_neg)
f1 = 2*precision*recall/(precision+recall)

accuracy = correct/(len(testPredict))
print("Correct: ", correct)
print("Incorrect: ", incorrect)
print("Total Prediction Accuracy: ", accuracy)
print("F1: ", f1)
print("Positive Predictive Value: ", precision)
print("True Positive Rate: ", recall)

days = list(range(len(testPredict)))
plt.plot(days, testY, label='sp500')
plt.plot(days, testPredict, label='predicted')
plt.legend(loc="upper left")
plt.show()
