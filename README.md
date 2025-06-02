# bitcoin-lstm-price-prediction
A deep learning project using LSTM neural networks with TensorFlow/Keras to predict future Bitcoin prices based on historical market data.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Objective](#objective)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

`bitcoin-lstm-price-prediction` leverages Long Short-Term Memory (LSTM) neural networks—implemented using TensorFlow and Keras—to forecast future Bitcoin prices from historical market data. LSTMs are a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies, making them ideal for time-series prediction tasks such as cryptocurrency price forecasting.

This project guides users through the process of collecting, preparing, and analyzing Bitcoin historical price data, building and training an LSTM model, evaluating its performance, and visualizing the predictions against actual market data. The project structure is modular, making it easy to adapt for other time-series forecasting tasks or cryptocurrencies.

---

## Features

- Automated retrieval and cleaning of historical Bitcoin price data using popular APIs.
- Robust data preprocessing and feature engineering pipeline.
- Configurable LSTM model architecture for time-series forecasting.
- Model training, validation, and performance evaluation.
- Visualization tools to compare predicted prices with actual market data.
- Modular code structure for extensibility and reuse.
- Reproducible experiments with configurable parameters.

---

## Objective

1. Collect historical Bitcoin price data from reliable APIs.
2. Preprocess and prepare the data for model training and validation.
3. Build and train an LSTM neural network to predict future Bitcoin prices.
4. Evaluate the model’s accuracy and effectiveness using appropriate metrics.
5. Visualize predictions versus actual prices to assess model performance.

---

## Technologies Used

- **Python 3.7+**
- **TensorFlow & Keras** - Deep learning framework for building and training neural networks.
- **Pandas** - Data manipulation and analysis.
- **NumPy** - Numerical operations and array handling.
- **Matplotlib & Seaborn** - Data visualization.
- **Requests** - HTTP library for data retrieval.
- **Jupyter Notebook** - Interactive analysis and experimentation.

---

## Dataset

The dataset consists of historical Bitcoin price data, including:
- Date
- Open price
- High price
- Low price
- Close price
- Trading volume

Data can be sourced from reputable cryptocurrency APIs such as:
- [CoinGecko](https://coingecko.com/)
- [CoinMarketCap](https://coinmarketcap.com/)
- [Binance](https://binance.com/)

A sample CSV (`bitcoin_prices.csv`) is provided in the `data/` directory. If you wish to update or expand the dataset, use the retrieval scripts or adapt them for your preferred data source.

---

## Project Structure

```
bitcoin-lstm-price-prediction/
├── data/
│   └── bitcoin_prices.csv
├── models/
│   └── bitcoin_price_lstm_model.h5
├── notebooks/
│   └── bitcoin_lstm_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── lstm_model.py
│   └── visualization.py
├── .gitignore
├── requirements.txt
└── README.md
```

- **data/**: Contains raw and preprocessed datasets.
- **models/**: Stores trained model files.
- **notebooks/**: Jupyter notebooks for exploratory analysis and experimentation.
- **src/**: Source code for data preprocessing, model training, and visualization.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/mbuckner1/bitcoin-lstm-price-prediction.git
    cd bitcoin-lstm-price-prediction
    ```

2. **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Prepare the Data

Ensure that the dataset (`bitcoin_prices.csv`) is present in the `data/` directory. You can also run the data retrieval/preprocessing script:

```bash
python src/data_preprocessing.py
```

This script will clean, normalize, and split the data for training and validation.

### 2. Train the LSTM Model

You can use the provided Jupyter notebook for step-by-step experimentation:

```bash
jupyter notebook notebooks/bitcoin_lstm_analysis.ipynb
```

Or run the model training script directly:

```bash
python src/lstm_model.py
```

Model checkpoints and the final trained model will be saved in the `models/` directory.

### 3. Visualize Model Predictions

To compare the model’s predictions with actual Bitcoin prices, run:

```bash
python src/visualization.py
```

This will generate plots and save them to the `notebooks/` or `reports/` directory.

---

## Results

The LSTM model’s predictions are visualized against actual price data to show how well the model captures market trends. Key metrics such as RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) are reported to assess model accuracy. Example results and plots are available in the `notebooks/` and `reports/` directories.

---

## Visualization

Below is an example of the type of output you can expect from the visualization script:

![Example Prediction Plot](docs/prediction_example.png)

The plot shows the predicted prices (in orange) versus the actual prices (in blue), helping to visually assess forecasting accuracy.

---

## Troubleshooting

- **Data retrieval errors:**  
  Ensure you have an active internet connection and the API keys (if required) are set up correctly.
- **Dependency issues:**  
  Double-check your Python version and that all dependencies are installed via `requirements.txt`.
- **Memory errors:**  
  Reduce batch size or sequence length in the configuration if you encounter memory issues.
- **Model not learning:**  
  Experiment with model hyperparameters, learning rate, or try more epochs.

---

## FAQ

**Q: Can I use this project for predicting other cryptocurrencies?**  
A: Yes! Adapt the data retrieval and preprocessing scripts for any time-series data.

**Q: Does this model guarantee profitable trading?**  
A: No. The model is for research and education only. Use at your own risk.

**Q: How do I improve model accuracy?**  
A: Try tuning the model’s hyperparameters, adding more features, or applying more advanced architectures.

---

## Contributing

Contributions are welcome! Please open issues or pull requests to suggest improvements, add features, fix bugs, or share your results. For larger contributions, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [CoinGecko API](https://coingecko.com/)
- [Bitcoin Historical Data](https://www.cryptodatadownload.com/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

*This repository is for educational and research purposes only. Cryptocurrency trading is risky and predictions made by machine learning models are not guarantees of future performance.*
