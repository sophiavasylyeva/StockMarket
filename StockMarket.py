# Recurrent Neural Networks and Stock Price Prediciton

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import tensorflow as tf
nltk.download('vader_lexicon')

class StockPricePrediction:
    def __init__(self, ticker, start_date='2010-01-01', end_date='2024-12-31'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.lstm_model = None
        self.gru_model = None


    # Fetch stock data
    def get_stock_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Data for {self.ticker} fetched successfully.")


    # Preprocess data
    def preprocess_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = self.scaler.fit_transform(self.data[['Close', 'Volume']])
        X, y = self.create_dataset(data_scaled)
        self.X_train, self.X_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
        self.y_train, self.y_test = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]
        print("Data preprocessing completed.")


    # Create dataset with 60-day window
    def create_dataset(self, data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i, 0])  # Using 'Close' price for X
            y.append(data[i, 0])  # Using 'Close' price for y
        return np.array(X), np.array(y)


    # Build LSTM model
    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_model = model
        print("LSTM model built successfully.")


    # Build GRU model
    def build_gru_model(self):
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.gru_model = model
        print("GRU model built successfully.")

    def train_models(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{self.ticker}_best_model.keras',  # Changed from .h5 to .keras
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        self.lstm_model.fit(
            self.X_train, 
            self.y_train, 
            epochs=20, 
            validation_split=0.1,
            batch_size=32,
            callbacks=callbacks
        )

        self.gru_model.fit(
            self.X_train, 
            self.y_train, 
            epochs=20, 
            validation_split=0.1,
            batch_size=32,
            callbacks=callbacks
        )


    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, predictions))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        return {'rmse': rmse, 'mape': mape}



    # Perform sentiment analysis on stock news
    def get_sentiment(self, news):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = []
       
        # Analyze the sentiment of each article
        for article in news:
            score = sid.polarity_scores(article)['compound']
            # Only add sentiment scores that are either negative or positive (exclude neutral scores)
            if score != 0:
                sentiment_scores.append(score)


        # Return the mean sentiment score
        if sentiment_scores:
            return np.mean(sentiment_scores)
        else:
            return 0  # If all articles were neutral, return 0


    def plot_predictions(self, model, title):
        predictions = model.predict(self.X_test)
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_test, label="Actual", color='blue')
        plt.plot(predictions, label="Predicted", color='red')

        # Add confidence interval
        plt.fill_between(
            range(len(predictions)),  # x-axis points
            predictions.flatten() - predictions.std(),  # lower bound
            predictions.flatten() + predictions.std(),  # upper bound
            alpha=0.2,  # transparency
            color='gray'
        )

        # Add labels and title
        plt.title(f'{title} Model: Actual vs Predicted Stock Prices', fontsize=14, pad=20)
        plt.xlabel('Time (Days)', fontsize=12)
        plt.ylabel('Stock Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.show()


    # Perform the entire analysis and visualization
    def perform_analysis(self):
        self.get_stock_data()
        self.preprocess_data()
        self.build_lstm_model()
        self.build_gru_model()
        self.train_models()


        # Evaluate performance
        lstm_rmse = self.evaluate_model(self.lstm_model, self.X_test, self.y_test)
        gru_rmse = self.evaluate_model(self.gru_model, self.X_test, self.y_test)


        print(f'LSTM RMSE: {lstm_rmse}')
        print(f'GRU RMSE: {gru_rmse}')


        # Visualize predictions
        self.plot_predictions(self.lstm_model, 'LSTM')
        self.plot_predictions(self.gru_model, 'GRU')


        # Example sentiment analysis on dummy news
        stock_news = [
            "Apple's stock hits new highs",
            "CEO of Apple announces major product",
            "Watch These Apple Price Levels After Stock Set a New Record High",
            "Apple Inc. (AAPL) is Attracting Investor Attention: Here is What You Should Know"
        ]
        sentiment_score = self.get_sentiment(stock_news)
        print(f"Sentiment score: {sentiment_score}")
        
        # Example usage:
if __name__ == "__main__":
    ticker = 'AAPL'  # Example: Apple stock
    stock_predictor = StockPricePrediction(ticker)
    stock_predictor.perform_analysis()
    
