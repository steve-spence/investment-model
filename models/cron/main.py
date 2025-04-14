from news_updaters import yahoo_news_source, saver, get_prices
from transformers import pipeline
import torch
from tqdm import tqdm


def __remove_bad_entries(news: dict):
    """
    Remove news entries that do not have a 'published' key
    """
    for symbol, news_list in news.items():
        for news_source in news_list:
            if 'posting_price' not in news_source:
                news_list.remove(news_source)

def update_news(news_dir="../data/old_news"):
    finbert = pipeline("text-classification", model="ProsusAI/finbert")

    yahoo = yahoo_news_source.YahooNews('accepted_stocks.json')

    manager = saver.StockNewsManager(news_dir)
    
    yahoo_news = yahoo.get_news('AAPL')

    all_news = [yahoo_news] # when we add more news sources they will go here

    # get prices for news articles
    for news in all_news:
        get_prices.load_stock_data(finbert_pipe=finbert, news=news, accepted_stocks=yahoo.get_accepted_stocks())
        __remove_bad_entries(news=news)

    for news_data in all_news:
        for symbol, news_list in tqdm(news_data.items(), desc='writing price data to file'):
            manager.save_news(symbol, news_list)

def update_model():
    pass

def main():
    update_news(news_dir='../data/news/')
    update_model()

    
if __name__=="__main__":
    main()