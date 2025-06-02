import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = os.path.join("..", "data", "bitcoin_prices.csv")
OUTPUT_PATH = os.path.join("..", "data", "bitcoin_prices_scaled.csv")
WINDOW_SIZE = 60

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date")
    return df

def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df["close"].values.reshape(-1, 1))
    return scaled, scaler

def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return X, y

def main():
    df = load_data()
    scaled, scaler = normalize_data(df)
    X, y = create_sequences(scaled, WINDOW_SIZE)
    # Save preprocessed data for model training
    pd.DataFrame({
        "X": [x.tolist() for x in X],
        "y": y
    }).to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
