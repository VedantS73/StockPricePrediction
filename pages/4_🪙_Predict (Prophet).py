import streamlit as st
from stock_data import stock_names_symbols
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon=""
)

symbol = None

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

if selected_stock:
    symbol = selected_stock.split(" ")[-1][1:-1]  # Extract symbol from selected stock

    # Load stock data (cached for better performance)
    data = load_stock_data(symbol)
    
    # Show and plot forecast
    st.subheader('Stock Data')
    st.write(data.head())
    
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_raw_data()
    
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365
    
    train_button = st.button("Predict Stock Movement with LSTM", key="train_button", help="Click to train")

if selected_stock and train_button:
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)