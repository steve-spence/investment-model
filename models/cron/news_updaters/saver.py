import json
import os

class StockNewsManager:
    def __init__(self, directory='news'):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    def load_news(self, symbol: str):
        file_path = os.path.join(self.directory, f"{symbol}.json")
        with open(file_path, 'r') as file:
            news_data = json.load(file)
        return news_data

    def save_news(self, symbol: str, news_list: list):
        file_path = os.path.join(self.directory, f"{symbol}.json")

        # Load existing news if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_news = json.load(file)
        else:
            existing_news = []

        # Append new news items
        existing_news.extend(news_list)

        # Save updated news list back to json file
        with open(file_path, 'w') as file:
            json.dump(existing_news, file, indent=2)

        #print(f"News saved successfully for {symbol}!")

