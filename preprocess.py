import pandas as pd

def preprocess(weather_tracker):
    df = pd.read_csv('./data/HistoricalPrices.csv')

    # retain only date and closing price. Convert Date to datetime
    df = df[['Date', ' Close']]
    df['Date'] = pd.to_datetime(df.Date)

    # reverse stock prices dataframe
    df = df.loc[::-1]

    weather_df = pd.read_csv('./data/new_york_city.csv')

    # retain only date and cloud cover for this test. Convert Date to datetime
    # cloudcover
    weather_df = weather_df[['date_time', weather_tracker]]
    weather_df.columns = ['Date', weather_tracker]
    weather_df['Date'] = pd.to_datetime(weather_df.Date)

    # merge the dataframes of stock and weather data
    df = pd.merge(df, weather_df, on="Date", how="right")

    # drop rows with NaN
    df = df.dropna()

    # get changes in stock price into dataframe and reduce it to the fields we want.
    df['diffs'] = df[' Close'].sub(df[' Close'].shift()).div(df[' Close'] - 1).fillna(0)
    df = df[[weather_tracker, 'diffs']]

    return df.values
