import json
from abc import abstractmethod

class NewsSource:
    """
    Abstract class for fetching news for a given stock symbol
    """
    def __init__(self, accepted_stocks_location):
        self.accepted_stocks = {}
        self.accepted_stocks_path = accepted_stocks_location

    def _load_stocks(self) -> None:
        """
        Load the accepted_stocks.json file
        """
        with open (self.accepted_stocks_path) as f: # This is relative to main.py I think
            self.accepted_stocks = json.load(f)

        # make key the symbol
        self.accepted_stocks = {item.pop('symbol'): item for item in self.accepted_stocks}

    def _load_news(self, symbol: str) -> list:
        """
        Return a list of news for a given stock symbol
        """
        # make sure we accept the symbol
        if not self.accepted_stocks[symbol]:
            raise Exception("Stock symbol not found in accepted_stocks.json")

    def get_accepted_stocks(self) -> dict:
        """
        Return the accepted_stocks dictionary
        """
        return self.accepted_stocks

    @abstractmethod
    def get_news(self, *args, **kwargs) -> dict:
        """
        Subclasses should implement their logic to fetch news (possibly using a shared approach).
        Return format: {symbol: [news_items]}
        """
        pass
