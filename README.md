# Financial Volatility Forecasting Using Stacked LSTMs
Javier Martín Nieto

## Overview

This project explores the use of deep learning for time series forecasting in financial risk management.

The main objective is to investigate whether a stacked Long Short-Term Memory (LSTM) neural network can accurately forecast the 30-day annualized volatility of Banco Santander (`SAN.MC`) using historical market data and technical indicators. The results support this hypothesis, with the model effectively capturing volatility regimes and outperforming a standard Naive (persistence) baseline.

## Dataset

The project utilizes historical market data downloaded via `yfinance` spanning from 2018 to 2026.

The final dataset contains:

- 2,000+ daily trading observations
- 8 input features including market indices (`SAN.MC`, `^VIX`, `^GSPC`), returns, and technical indicators
- 10-day historical volatility (`SAN_Vol_10d`)
- 14-day Relative Strength Index (`SAN_RSI_14d`)
- A continuous target variable representing the 30-day annualized volatility (`Target_Vol_30d`)

The data is split chronologically into Train (60%), Validation (20%), and Test (20%) sets to prevent temporal data leakage.

## Model Architecture

The architecture processes sequences of 60 trading days across 8 features to predict the volatility of the following day:

R^(60x8) -> LSTM(60) -> Dropout(0.2) -> LSTM(30) -> Dropout(0.2) -> Dense(15) -> Dense(1)

The model is trained using Mean Squared Error (MSE) as the loss function with the Adam optimizer (`learning_rate = 0.001`), incorporating Dropout regularization to prevent overfitting and Early Stopping to determine the optimal training epochs.

The underlying idea is:

- Sequence Learning -> Capture long-term dependencies and non-linear interactions across macroeconomic and technical features
- Normalized Scale -> Training performed on scaled data ($[0, 1]$) with MinMax scaling fit strictly on the training partition
- Inverse Transformation -> Evaluation conducted on real percentage volatility units (%) for financial interpretability

**Technologies used**
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Yahoo Finance API (`yfinance`)
- Jupyter Notebook

## Methodology

- Market data extraction and daily forward-fill processing for holiday alignment
- Logarithmic return calculations and rolling window feature engineering (10-day volatility, 30-day target volatility, and 14-day RSI)
- Strict chronological dataset split (60% Train, 20% Validation, 20% Test) to avoid look-ahead bias
- Feature scaling using `MinMaxScaler` fitted exclusively on the training set
- Multi-run hyperparameter estimation to dynamically select optimal training epochs based on validation loss progression using median stopping
- Model retraining on the combined Train + Validation partition (80%) using the computed optimal epochs
- Evaluation on the unseen Test set (20%) in real percentage scale using RMSE, MAE, and MAPE
- Benchmark comparison against a Naive Persistence Baseline ($\hat{y}_t = y_{t-1}$)
- Diagnostic residual analysis to verify unbiased model predictions

## Results

- The stacked LSTM model successfully learns non-linear volatility dynamics, achieving lower prediction error compared to simple persistence strategies.
- The model maintains a low Mean Absolute Error (MAE) on the test set, demonstrating accurate tracking of volatility shifts without severe lag.
- Residual analysis confirms that errors are centered around zero, indicating an unbiased estimation across different market regimes.

## Limitations & Future Work

- The current feature set relies on daily close prices; incorporating high-frequency intraday data (e.g., realized volatility from order books) could enhance predictive accuracy.
- Hyperparameter tuning could be expanded using automated Bayesian Optimization (e.g., KerasTuner) to explore deeper architectures and learning rate schedules.
- Future iterations could explore hybrid models (e.g., CNN-LSTM or Transformer-based architectures) to capture both spatial feature interactions and longer temporal horizons.
