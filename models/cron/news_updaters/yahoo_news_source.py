from datetime import datetime
from typing import Optional
from tqdm import tqdm
import feedparser
from news_updaters import news_source

class YahooNews(news_source.NewsSource):
    def _load_news(self, symbol: str):
        """
        Return a list of news for a given stock symbol
        """
        super()._load_news(symbol)

        # make sure we accept the symbol
        if not self.accepted_stocks[symbol]:
            return "Stock symbol not found in accepted_stocks.json"

        # get the security name
        security = self.accepted_stocks[symbol]['security'].lower().strip()
        rss_url = f"https://finance.yahoo.com/rss/2.0/headline?s={symbol}"

        feed = feedparser.parse(rss_url)

        news_list = []
        for i , entry in enumerate(feed.entries):
            if security in entry.summary.lower(): # Todo: play around with not in vs in
                continue
            news_data = {
                'title': entry.title,
                'published': int(datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z').timestamp())  # unix timestamp
            }
            news_list.append(news_data)

        return news_list # [{title: 'title', published: 'published'}]

    def get_news(self, symbol: Optional[str] = None) -> dict:
        """
        Get news for all data in accepted_stocks.json
        If you pass in a symbol it will only update news for that specific symbol
        """
        super()._load_stocks()

        if symbol:
            return {symbol: self._load_news(symbol)}

        news_data = {}
        for symbol in tqdm(self.accepted_stocks.keys(), total=len(self.accepted_stocks.keys()), desc='Getting recent yahoo news'):
            news_data[symbol] = self._load_news(symbol)
        return news_data


