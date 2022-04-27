# Stock Prediction Project
In this project, we attempt to predict the stock market using historical stock market and weather data.

## Installing required libraries
Installing the required libraries is as simple as using the requirements file and getting them from pip:

```console
$ pip install -r requirements.txt
```

## Retrieving historical data
We used two sources for data, yahoo finance for historical stock data and world weather online for historical weather data.

To get stock data downloaded locally into a handful of csv files, run:

```console
python getStockData.py
```

To get weather data downloaded locally into a csv file, run:

```console
python getWeatherData.py
```

The downloaded csv files are required for running the prediction models. You may also change the location or dates of the
weather data by going into `getWeatherData.py` and modifying the location or dates on lines 12 and 9 respectively.

## Running the prediction model
There are a few options required to run the prediction model. First, you should decide if you want to use the Recurrent Neural Network
or Long Short-Term Memory model. Then specify which weather data to help train the model with. The options are `cloudcover, sunHour, maxtempC`.
Here is an example using RNN and cloud cover data:

```console
$ python runModel.py -m RNN -w cloudcover
```

This will start training your model using the historical stock and weather data that you previously downloaded. After it runs the prediction,
it will provide the run's evaluation metrics and a graph of the prediction vs actual prices.

You can see the full usage here:

```console
$ python runModel.py -h
usage: runModel.py [-h] -m {RNN,LSTM} -w {cloudcover,sunHour,maxtempC}

Run the stock prediction model with certain inputs depending on what arguments you provide

optional arguments:
  -h, --help            show this help message and exit
  -m {RNN,LSTM}, --model-type {RNN,LSTM}
                        Run the model using an RNN or LSTM.
  -w {cloudcover,sunHour,maxtempC}, --weather-option {cloudcover,sunHour,maxtempC}
                        Run the model using the weather option you desire.
```

### Here are some sources we used during this project
- https://www.youtube.com/watch?v=3eZ56HERVbk
- https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
- https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
- https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
