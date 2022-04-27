# from this tutorial: https://www.youtube.com/watch?v=K_JQlIDzBpY
import pandas as pd
import yfinance as yf

sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp_wiki_df_list = pd.read_html(sp_wiki_url)
sp_df = sp_wiki_df_list[0]
sp_tickers = list(sp_df['Symbol'].values)
df = yf.download(sp_tickers)

df['Adj Close'].to_csv("./data/adjClose.csv")
df['Close'].to_csv("./data/close.csv")
df['High'].to_csv("./data/high.csv")
df['Low'].to_csv("./data/low.csv")
df['Open'].to_csv("./data/open.csv")
df['Volume'].to_csv("./data/volume.csv")
