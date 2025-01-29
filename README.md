# Interactive Stock Analysis Application

This Streamlit application provides an interactive way to analyze historical stock data, visualize trends, and perform basic predictive analysis using an LSTM model.

## Features

*   **Data Retrieval:** Fetches historical stock data directly from Yahoo Finance using the `yfinance` library.
*   **Summary Statistics:** Displays a summary of key statistics (mean, std, min, max, etc.) for the fetched stock data.
*   **General Information:** Shows general information about the dataset, including data types and non-null counts.
*   **Interactive Visualizations:**
    *   Line chart of the closing price over the past year.
    *   Line chart of trading volume over the past year.
    *   Scatter plot of risk vs. expected volume.
*   **LSTM-Based Prediction:** Uses a Long Short-Term Memory (LSTM) neural network to predict the future close price of the stock based on historical data.
    *   Displays Root Mean Squared Error (RMSE) to evaluate the model performance.
    *   Displays a chart comparing training, validation, and predicted prices
    *   Displays a data frame of predicted values with the original values for comparison
*   **Interactive Interface:**
    *   Uses Streamlit for an easy-to-use web interface.
    *   Allows users to enter a stock ticker symbol via a text input in the sidebar.

## How to Use

1.  **Clone the repository:** (If you have a repository for this code)

    ```bash
    git clone [repository_link]
    cd [repository_directory]
    ```

2.  **Install dependencies:**

    ```bash
    pip install streamlit pandas numpy matplotlib seaborn yfinance scikit-learn tensorflow keras
    ```

3.  **Run the application:**

    ```bash
    streamlit run stock_app.py
    ```

4.  **Access in browser:** Streamlit will automatically open the app in your default web browser.

5.  **Enter stock ticker:** In the sidebar, enter the ticker symbol of the stock you wish to analyze (e.g., `AAPL`, `GOOG`, `TSLA`, `TCS.NS`).

6.  **View results:** The app will then display the analysis, plots, and predictions for the chosen stock.

## Technologies Used

*   **Python:** The core programming language.
*   **Streamlit:** Used to build the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical computations.
*   **Matplotlib:** For creating static visualizations.
*   **Seaborn:** For enhancing visualizations.
*   **yfinance:** Used to fetch financial data from Yahoo Finance.
*   **scikit-learn:** Used for scaling data.
*   **TensorFlow and Keras:** Used to build and train the LSTM model.

## Disclaimer

This application is intended for educational purposes and basic analysis only. The predictive analysis is based on a simple LSTM model, and its predictions should not be used for financial investment decisions. Always consult a professional financial advisor before making any investment.
