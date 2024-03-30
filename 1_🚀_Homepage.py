import streamlit as st
from stock_data import stock_names_symbols
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon=""
)

st.title("Stock Price Prediction")
st.sidebar.success("Select a page above.")

@st.cache_data
def load_stock_data(symbol):
    """
    Function to load stock data from CSV file based on symbol.
    """
    file_path = f"data/{symbol}.csv"
    return pd.read_csv(file_path)

# Dropdown with options
selected_stock = st.selectbox(
    "Select a stock:",
    [f"{stock_name} ({symbol})" for stock_name, symbol in stock_names_symbols.items()]
)

time_range = st.selectbox(
    "Select time range:",
    ["7d", "30d", "6m", "1y", "5y", "All"]
)

if selected_stock:
    symbol = selected_stock.split(" ")[-1][1:-1]  # Extract symbol from selected stock

    # Load stock data (cached for better performance)
    stock_data = load_stock_data(symbol)

    # Filter stock data based on selected time range
    if time_range == "7d":
        stock_data = stock_data[-7:]  # Select data for the last 7 days
    elif time_range == "30d":
        stock_data = stock_data[-30:]  # Select data for the last 30 days
    elif time_range == "6m":
        # Select data for the last 6 months (assuming each row represents one day)
        stock_data = stock_data[-6*30:]
    elif time_range == "1y":
        # Select data for the last year (assuming each row represents one day)
        stock_data = stock_data[-365:]
    elif time_range == "5y":
        # Select data for the last 5 years (assuming each row represents one day)
        stock_data = stock_data[-5*365:]
    elif time_range == "All":
        # No filtering, show all data
        pass

    # Plot stock data
    fig = px.line(stock_data, x='Date', y='Close', title=f"Stock Chart for {selected_stock} ({time_range})")
    st.plotly_chart(fig)
