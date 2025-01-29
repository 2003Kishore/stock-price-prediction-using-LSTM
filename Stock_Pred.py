import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from pandas_datareader import data as pdr
import yfinance as yf

pdr.override = yf.download # Call it right after importing from pandas_datareader

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


def analyze_stock(stock_ticker):
    """Analyzes stock data for a given ticker symbol."""

    # Set up End and Start times for data grab
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    try:
        # Fetch stock data
        stock_data = yf.download(stock_ticker, start, end)
        if stock_data.empty:
             st.error(f"No data found for ticker: {stock_ticker}")
             return None
    except Exception as e:
        st.error(f"Error fetching data for ticker {stock_ticker}: {e}")
        return None

    # Assign company name (for plotting)
    stock_data["company_name"] = stock_ticker
    
    # Summary Stats
    st.subheader(f"Summary Statistics for {stock_ticker}")
    st.dataframe(stock_data.describe())

    # General info
    st.subheader(f"Information for {stock_ticker}")
    st.text(stock_data.info())

    # Extract 'Close' and 'Volume' data
    close_data = stock_data[('Close', stock_ticker)]
    volume_data = stock_data[('Volume', stock_ticker)]

    # Plot Closing Price
    st.subheader(f"Closing Price of {stock_ticker}")
    fig_close, ax_close = plt.subplots(figsize=(10, 6))
    close_data.plot(ax=ax_close)
    ax_close.set_ylabel('Close Price')
    ax_close.set_xlabel(None)
    ax_close.set_title(f"Closing Price of {stock_ticker}")
    st.pyplot(fig_close)

    # Plot Trading Volume
    st.subheader(f"Sales Volume for {stock_ticker}")
    fig_volume, ax_volume = plt.subplots(figsize=(10, 6))
    volume_data.plot(ax=ax_volume)
    ax_volume.set_ylabel('Volume')
    ax_volume.set_xlabel(None)
    ax_volume.set_title(f"Sales Volume for {stock_ticker}")
    st.pyplot(fig_volume)

      # Risk Calculation (Using Volume)
    st.subheader(f'Risk vs Expected Volume for {stock_ticker}')
    area = np.pi * 20
    fig_risk, ax_risk = plt.subplots(figsize=(8,6))
    ax_risk.scatter(volume_data.mean(), volume_data.std(), s=area)
    ax_risk.set_xlabel('Mean Volume')
    ax_risk.set_ylabel('Volume std')
    ax_risk.annotate(stock_ticker, xy=(volume_data.mean(), volume_data.std()), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
                     arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
    ax_risk.set_title(f'Risk vs Expected Volume for {stock_ticker}')
    st.pyplot(fig_risk)
    
        
    #Predictive Analysis
    st.subheader(f"Predictive Analysis for {stock_ticker}")
    
    # Data Preparation
    data = close_data.to_frame()
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # LSTM Model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1, verbose = 0)
    
    # Testing Data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Model Predictions
    predictions = model.predict(x_test, verbose = 0)
    predictions = scaler.inverse_transform(predictions)
    
    # Model Evaluation
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Plot Predictions
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    st.subheader(f'Model for {stock_ticker}')
    fig_pred, ax_pred = plt.subplots(figsize=(10,6))
    ax_pred.set_xlabel('Date')
    ax_pred.set_ylabel('Close Price USD ($)')
    ax_pred.plot(train['Close'])
    ax_pred.plot(valid[['Close', 'Predictions']])
    ax_pred.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    ax_pred.set_title(f'Model for {stock_ticker}')
    st.pyplot(fig_pred)
    st.write("\nPredictions:")
    st.dataframe(valid)

    return stock_data

def main():
    st.title("Stock Analysis App")
    st.sidebar.header("Enter Stock Ticker")
    stock_ticker = st.sidebar.text_input("Ticker Symbol (e.g., AAPL, TCS.NS)", "AAPL").upper()
    
    if stock_ticker:
        stock_data = analyze_stock(stock_ticker)

if __name__ == "__main__":
    main()