import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

DATA_PATH = os.path.join("..", "data", "bitcoin_prices_scaled.csv")
MODEL_PATH = os.path.join("..", "models", "bitcoin_price_lstm_model.h5")
WINDOW_SIZE = 60

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = np.array([eval(x) for x in df["X"]])
    y = np.array(df["y"])
    return X, y

def build_model(window_size=WINDOW_SIZE):
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    X, y = load_data()
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Simple train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(WINDOW_SIZE)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
