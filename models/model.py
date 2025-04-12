import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTM_Model:
    def __init__(self, num_time_steps=50, features=None, news_dir='../data/news/'):
        self.time_steps = num_time_steps
        self.stock_data = {}
        self.model = None
        self.X = None
        self.y = None
        self.news_dir = news_dir
        self.feature_dim = 8  # 7 prices + 1 sentiment
        self.bad_entries = 0

    def create_sample_data(self, total_entries=60):
        """Create sample data for testing purposes"""
        sample_data = []
        for i in range(total_entries):
            sample_data.append({
                "day_prices": {      
                    "13:30": 106.11 + np.random.normal(0, 1),
                    "14:30": 105.71 + np.random.normal(0, 1),
                    "15:30": 105.53 + np.random.normal(0, 1),
                    "16:30": 105.29 + np.random.normal(0, 1),
                    "17:30": 105.41 + np.random.normal(0, 1),
                    "18:30": 105.74 + np.random.normal(0, 1),
                    "19:30": 105.35 + np.random.normal(0, 1)
                },
                "sentiment": str(0.5 + np.random.rand()/2),
                "published": f"2024-05-{str(6 + i).zfill(2)}"
            })
        self.stock_data['SAMPLE'] = pd.DataFrame(sample_data)

    def load_all_data(self):
        """Load all JSON files from news directory"""
        if not os.path.exists(self.news_dir):
            raise FileNotFoundError(f"News directory not found: {self.news_dir}")
            
        for filename in os.listdir(self.news_dir):
            symbol = os.path.splitext(filename)[0]
            file_path = os.path.join(self.news_dir, filename)
            self.stock_data[symbol] = pd.read_json(file_path)

    def create_model(self):
        """Create LSTM model architecture"""
        inputs = Input(shape=(self.time_steps, self.feature_dim))
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )

    def _process_data(self):
        """Internal method to process raw data into sequences"""
        all_sequences = []
        all_labels = []
        
        for symbol, df in self.stock_data.items():
            # soft all entires in df by published and get rid of old indexes and make new ones
            df = df.sort_values('published').reset_index(drop=True) 

            feature_vectors = []
            
            # make feature vectors
            for _, row in df.iterrows():
                try:
                    # Process prices (throws error cuz row['day_prices'] is a float somehow)
                    prices = row['day_prices'].items()

                    # bad entry with less than 7 time entries
                    if len(prices) != 7: #-----------------------
                        self.bad_entries += 1
                        continue
                except Exception:
                    self.bad_entries += 1
                    continue


                prices = [p[1] for p in prices] # remove dates and just get prices

                # make sure sentiment is a float and concat with feature vec
                sentiment = float(row['sentiment'])
                feature_vectors.append(prices + [sentiment])
            
            # learn from the features in range of the time step
            for i in range(len(feature_vectors) - self.time_steps):
                try:
                    sequence = feature_vectors[i:i+self.time_steps] # todo : print these out and make sure the values are right
                    next_day_close = feature_vectors[i+self.time_steps][6]
                    current_close = sequence[-1][6]
                    #print(current_close - next_day_close)
                    all_sequences.append(sequence)
                    all_labels.append(1 if next_day_close > current_close else 0)
                except IndexError:
                    self.bad_entries += 1

        self.X = np.array(all_sequences)
        self.y = np.array(all_labels)
        
        # Add some noise to fix sample data
        if 'SAMPLE' in self.stock_data:
            self.y = self.y ^ np.random.randint(0, 2, size=len(self.y))

    def fit(self, epochs=10, batch_size=32, validation_split=0.2):
        """Train the model"""
        if self.X is None:
            self._process_data()
            
        self.model.fit(
            self.X, self.y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True
        )

    def predict(self, symbol: str):
        """Make prediction for a specific stock"""
        if symbol not in self.stock_data:
            raise ValueError(f"No data available for {symbol}")
            
        df = self.stock_data[symbol].sort_values('published')
        feature_vectors = []
        
        # Process historical data
        for _, row in df.iterrows():
            prices = sorted(row['day_prices'].items())
            prices = [p[1] for p in prices]
            sentiment = float(row['sentiment'])
            feature_vectors.append(prices + [sentiment])
        
        if len(feature_vectors) < self.time_steps:
            raise ValueError(f"Insufficient data for {symbol}. Need {self.time_steps} days.")
        
        # Use most recent sequence
        sequence = np.array([feature_vectors[-self.time_steps:]])
        return self.model.predict(sequence)[0][0]

    def evaluate(self, test_dir=None):
        """Evaluate model performance"""
        if test_dir:
            original_data = self.stock_data
            self.stock_data = {}
            self.load_all_data(test_dir)
            
        if self.X is None:
            self._process_data()
            
        return self.model.evaluate(self.X, self.y)
    
    def save_model(self, filepath='model/lstm_model.keras'):
        """Save the trained model to disk"""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model/lstm_model.keras'):

        """Load a pre-trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at {filepath}")
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        

if __name__ == "__main__":
    """"
    do you want to run the saved model?
    """
    isSaved = False
    
    model = LSTM_Model(num_time_steps=30)

    if isSaved:
        # then load it
        model.load_model('model/lsmt_model.keras')
    else:
        # then train it
        model.load_all_data()
        model.create_model()
        model.fit(epochs=4, batch_size=32, validation_split=0.2)
        model.save_model()
     
    # todo : fix the save and load im tired im going to bed

    try:
        prediction = model.predict('AAPL')
        print(f"Predicted price movement: {'Up' if prediction > 0.5 else 'Down'}")
    except Exception as e:
        print(f"Prediction error: {e}")