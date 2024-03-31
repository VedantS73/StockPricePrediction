import streamlit as st
from stock_data import stock_names_symbols
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon=""
)

symbol = None

st.title("Stock Price Prediction (Intraday)")
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

if selected_stock:
    symbol = selected_stock.split(" ")[-1][1:-1]  # Extract symbol from selected stock

    # Load stock data (cached for better performance)
    stock_data = load_stock_data(symbol)
    
    # Plot stock data
    fig = px.line(stock_data, x='Date', y='Close', title=f"Stock Chart for {selected_stock}")
    st.plotly_chart(fig)
    
    train_button = st.button("Predict Stock Movement with LSTM", key="train_button", help="Click to train")

if selected_stock and train_button:
    # Perform training
    pass