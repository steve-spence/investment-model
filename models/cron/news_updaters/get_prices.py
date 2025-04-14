from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd


def load_stock_data(finbert_pipe, news: list, accepted_stocks: dict, day_forecast=7):
    """
    Save the price data for all news data for each stock symbol
    param finbert_pipe is a financial bert model from huggingface
    param news list of news data
    param accepted_stocks dictionary of accepted stocks
    param day_forecast number of days to forecast ahead if no data for that stock symbol on that date
    """
    rounding_precision = 8

    # Get the price data for each symbol in the news directory
    for symbol in tqdm(accepted_stocks.keys(), desc='Getting price data for stocks'):
        if symbol not in news:
            continue
        recorded_dates = {} # list of dates we have already recorded stock data for
        for news_source in news[symbol]: # {'title':, 'published':}
            # Get the stock price at the time of the news

            # Get the dates for the stock data
            # We get current day and next because you cant input same day for both start and end
            news_obj = datetime.fromtimestamp(news_source['published'], tz=timezone.utc)
            start_date = str(news_obj).split(' ')[0]
            end_date = str(news_obj + timedelta(days=day_forecast)).split(' ')[0] # offset by day_forecast the amount of days to store

            # Check if we have data for that date
            if start_date not in recorded_dates:
                try: # Check if we have data for that date
                    yf_data = yf.download(tickers=symbol, start=start_date, interval='1h', end=end_date, timeout=None, progress=False)
                    yf_data.index = pd.to_datetime(yf_data.index)
                    for i in range(day_forecast):
                        # the next dates of the news
                        current_dt = news_obj + timedelta(days=i)
                        current_date = str(current_dt.date())
                        #print('cu', current_date)

                        day_data = yf_data[yf_data.index.date == current_dt.date()]

                        #print('day_data', day_data['Open'][symbol])
                        # Save the stock data for that date
                        recorded_dates[current_date] = {
                            'open': round(yf_data.iloc[i]['Open'].values[0], rounding_precision),
                            'close': round(yf_data.iloc[i]['Close'].values[-1], rounding_precision),
                            # zip datetime and prices
                            'prices': dict(
                                zip(
                                    [dt.isoformat() for dt in day_data['Open'][symbol].index.to_list()],
                                    day_data['Open'][symbol].to_list()
                                )
                              )
                            }

                        # if the next day is in the recorded_dates, break
                        if str(news_obj + timedelta(days=1)).split(' ')[0] in recorded_dates:
                            break

                except Exception as e: # index exception and api limit exception
                    print('exception', e)
                    news[symbol].remove(news_source)
                    continue

            day_prices = recorded_dates[start_date]['prices']
            if day_prices is None:
              news[symbol].remove(news_source)
            news_source['day_prices'] = day_prices
            news_source['sentiment'] = finbert_pipe(news_source['title'])[0]['score']


