# StockMarket

Recurrent Neural Network-Based Stock Price Prediction System

Project Overview:
    Our team aims to develop and implement an advanced stock price prediction system using Recurrent Neural Networks (RNNs), specifically comparing the performance of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures. The project will focus on predicting stock prices for different companies while incorporating multiple features including price history and trading volume data. 

Methodology
    Data Collection and Preprocessing
    - Utilize the finance API to gather historical stock data from 2010 to present
    - Implement data normalization using MinMaxScaler
    - Create sequential datasets with 60-day windows for prediction
Model Development
    - Implement two neural network architectures:
        - LSTM-based model with dual LSTM layers
        - GRU-based model with dual GRU layers
    - Enhance both models with volume data integration
    - Experiment with different activation functions (ReLU, sigmoid, tanh)
Model Optimization
    - Implement hyperparameter tuning
    - Optimize batch size and learning rate
    - Compare performance metrics between LSTM and GRU models
Sentiment Analysis
    - Implement sentiment analysis based on daily stock news
    - Compare performance based on accuracy of analysis

Deliverables
    Source Code
        - Complete Python implementation of both LSTM and GRU models
        - Data preprocessing and feature engineering scripts
        - Model evaluation and comparison scripts
    Documentation
        - Detailed technical documentation of model architectures
        - Performance analysis report
    Results and Analysis
        - Comparative analysis of LSTM vs GRU performance
        - Root Mean Square Error (RMSE) metrics for both models
    Interactive Visualization Dashboard
        - Web interface for real-time predictions
        - Interactive charts using Recharts library featuring:
            - Time series plots comparing actual vs predicted prices
            - Volume analysis visualization
            - Model performance comparison charts
