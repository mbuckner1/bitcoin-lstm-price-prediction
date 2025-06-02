import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = os.path.join("..", "data", "bitcoin_prices.csv")
SCALED_DATA_PATH = os.path.join("..", "data", "bitcoin_prices_scaled.csv")
MODEL_PATH = os.path.join("..", "models", "bitcoin_price_lstm_model.h5")
WINDOW_SIZE = 60

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")
    close_data = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    return df, scaled_data, scaler

def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def main():
    df, scaled_data, scaler = load_data()
    X, y = create_sequences(scaled_data, WINDOW_SIZE)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y[split:]

    model = load_model(MODEL_PATH)
    y_pred = model.predict(X_test)
    # Inverse scaling
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'][-len(y_test_inv):], y_test_inv, label='Actual Price')
    plt.plot(df['date'][-len(y_pred_inv):], y_pred_inv, label='Predicted Price')
    plt.title('Bitcoin Price Prediction - LSTM')
    plt.xlabel('Date')
    plt.ylabel('USD')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
