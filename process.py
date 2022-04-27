import math
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.callbacks import TensorBoard


def process(values, model_type='RNN'):
    n_train = math.floor(len(values)*3/4)
    train = values[:n_train, :]
    test = values[n_train:, :]
    trainX, trainY = _convertToMatrix(train, 5)
    testX, testY = _convertToMatrix(test, 5)

    # split into input and outputs
    trainY = train[5:, -1]
    testY = test[5:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = trainX.reshape((trainX.shape[0], 5, 2))
    test_X = testX.reshape((testX.shape[0], 5, 2))

    model = _getModel(train_X, model_type)

    # create these folders if they does not exist for logging
    if not os.path.isdir("results"):
        os.mkdir("results")
        if not os.path.isdir("logs"):
            os.mkdir("logs")

    # Create some callbacks for tracking
    tensorboard = TensorBoard(log_dir=os.path.join("logs", "stockPredictionLog"))

    # train the model
    model.fit(
        trainX,
        trainY,
        epochs=200,
        batch_size=64,
        validation_data=(testX, testY),
        callbacks=[tensorboard],
        verbose=2
    )

    # predict the price and plot
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # print our metrics
    _getMetrics(testPredict, testY)

    days = list(range(len(testPredict)))
    plt.plot(days, testY, label='sp500')
    plt.plot(days, testPredict, label='predicted')
    plt.legend(loc="upper left")
    plt.show()


def _convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)-step):
        d = i +step
        X.append(data[i:d])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


def _getModel(train_X, model_type='RNN'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=32, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
    else:
        model.add(SimpleRNN(units=32, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='huber_loss', metrics=['mean_absolute_error'], optimizer='adam')
    model.summary()
    return model


def _getMetrics(testPredict, testY):
    correct = incorrect = true_pos = true_neg = false_pos = false_neg = 0
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
