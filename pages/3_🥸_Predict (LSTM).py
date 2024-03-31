import streamlit as st
from stock_data import stock_names_symbols
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
    stock_data = load_stock_data(symbol)
    
    # Plot stock data
    fig = px.line(stock_data, x='Date', y='Close', title=f"Stock Chart for {selected_stock}")
    st.plotly_chart(fig)
    
    train_button = st.button("Predict Stock Movement with LSTM", key="train_button", help="Click to train")

if selected_stock and train_button:
    # Perform training
    st.info(f"Training process for {symbol} started...")
    df = load_stock_data(symbol)
    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    model.save(f"data/trained/{symbol}.h5")
    st.success(f"Training process for {symbol} completed successfully.")
    
    # Create the test data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model's predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Prepare the data for visualization
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Plot the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted'))
    fig.update_layout(
        title='Model',
        xaxis_title='Date',
        yaxis_title='Close Price USD ($)',
        legend=dict(x=0, y=1.0),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    st.plotly_chart(fig)

    # Show the table containing 'valid' data
    st.write("Valid and Preicted Prices:")
    st.write(valid)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(valid['Close'], valid['Predictions'])

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(valid['Close'], valid['Predictions'])

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Calculate absolute percentage error for each prediction
    valid['Absolute_Percentage_Error'] = abs((valid['Close'] - valid['Predictions']) / valid['Close']) * 100

    # Define threshold for considering predictions as correct
    threshold = 8  # 1% threshold

    # Count number of correct predictions
    correct_predictions = (valid['Absolute_Percentage_Error'] <= threshold).sum()

    # Total number of predictions
    total_predictions = len(valid)

    # Calculate percentage accuracy
    percentage_accuracy = (correct_predictions / total_predictions) * 100
    
    st.write(f"Percentage Accuracy: {percentage_accuracy:.2f}%")

    # Calculate overall performance and accuracy of the model (you can add your specific metrics here)
    # For example:
    # performance_metric = ...
    # accuracy_metric = ...

    # Display performance and accuracy
    # st.write(f"Performance Metric: {performance_metric}")
    # st.write(f"Accuracy Metric: {accuracy_metric}")
