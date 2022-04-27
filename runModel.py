import sys
import argparse
from preprocess import preprocess
from process import process

def runModel():
    desc = 'Run the stock prediction model with certain inputs depending on what arguments you provide'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-m', '--model-type', choices=['RNN', 'LSTM'], required=True, dest='model_type',
        help='Run the model using an RNN or LSTM.'
    )
    parser.add_argument(
        '-w', '--weather-option', choices=['cloudcover', 'sunHour', 'maxtempC'], required=True,
        dest='weather_option', help='Run the model using the weather option you desire.'
    )
    args = vars(parser.parse_args())

    values = preprocess(args['weather_option'])
    process(values, args['model_type'])

if __name__ == '__main__':
    runModel()
