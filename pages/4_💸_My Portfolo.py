import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon=""
)

st.title("My Portfolio")

st.markdown('### Overview')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Today's Growth","0.00 %")
col2.metric("Future Growth","30%")
col3.metric("Future Growth","30%")
col4.metric("Future Growth","30%")

st.markdown('---')

# Initialize portfolio DataFrame or retrieve from session state
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=['Stock', 'Quantity', 'Total Price'])

# Input form to add entries to portfolio
stock_name = st.text_input("Enter stock name:")
quantity = st.number_input("Enter quantity:", min_value=0, step=1)
total_price = st.number_input("Enter total price:", min_value=0.0, step=0.01)

if st.button("Add to Portfolio"):
    # Create a new row as a dictionary
    new_row = {'Stock': stock_name, 'Quantity': quantity, 'Total Price': total_price}

    # Concatenate the new row with the existing DataFrame only if it's not empty
    if not st.session_state.portfolio_df.empty:
        st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, pd.DataFrame.from_dict([new_row])], ignore_index=True)
    else:
        st.session_state.portfolio_df = pd.DataFrame.from_dict([new_row])

    st.success("Entry added to portfolio.")

# Display portfolio
st.markdown('### Portfolio')
st.table(st.session_state.portfolio_df)
