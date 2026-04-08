import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Gold Price Prediction", layout="centered")

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    try:
        with open('gold_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: 'gold_model.pkl' or 'scaler.pkl' not found. Please ensure they are in the same directory.")
        return None, None

model, scaler = load_assets()

# GUI Header
st.title("💰 Gold Price Prediction App")
st.markdown("""
Enter the market parameters below to predict the **Closing Price** of Gold.
""")

# Input Fields based on the features used in training
# Features: High, Low, Open, Volume
st.sidebar.header("Input Parameters")

def user_input_features():
    high = st.sidebar.number_input("High Price", min_value=0.0, value=2000.0, step=0.1)
    low = st.sidebar.number_input("Low Price", min_value=0.0, value=1950.0, step=0.1)
    open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=1975.0, step=0.1)
    volume = st.sidebar.number_input("Trading Volume", min_value=0, value=5000, step=1)
    
    data = {
        'High': high,
        'Low': low,
        'Open': open_price,
        'Volume': volume
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input parameters
st.subheader("Market Parameters Entered")
st.write(input_df)

if model and scaler:
    # Preprocessing the input
    try:
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        if st.button("Predict Gold Price"):
            prediction = model.predict(input_scaled)
            
            st.success(f"### Predicted Closing Price: ${prediction[0]:,.2f}")
            
            # Additional UI feedback
            st.info(f"Prediction based on {type(model).__name__} model.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model and Scaler assets are missing. Please upload the .pkl files.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit for Gold Market Analysis")