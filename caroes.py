import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Helper function to safely load pickle files
def load_pickle(filename):
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb"))
    else:
        st.warning(f"⚠ Missing file: {filename}. Please make sure it is in the same folder as this app.")
        return None

# Load saved artifacts
model = load_pickle("car_price_model.pkl")
scaler = load_pickle("scaler.pkl")
dummy_columns = load_pickle("dummy_columns.pkl")
manu_model_mapping = load_pickle("manu_model_mapping.pkl")

st.set_page_config(page_title="Car Price Prediction App", layout="centered")
st.title("🚗 Car Price Prediction")

st.markdown("Enter car details below to get the estimated price.")

# Only show form if model is available
if model and scaler and dummy_columns is not None and manu_model_mapping is not None:
    manufacturer = st.selectbox("Manufacturer", list(manu_model_mapping.keys()))
    model_selected = st.selectbox("Model", manu_model_mapping[manufacturer])
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1, value=2018)
    mileage = st.number_input("Mileage (km driven)", min_value=0, step=1000, value=50000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])
    engine_size = st.number_input("Engine Size (L)", min_value=0.8, max_value=6.0, step=0.1, value=1.5)

    if st.button("Predict Price"):
        # Create dataframe for single input with all dummy columns directly
        input_data = pd.DataFrame({
            "Year of manufacture": [year],
            "Engine size": [engine_size],
            "Mileage": [mileage],
            "Fuel type": [fuel_type],
            "Model": [model_selected]
        })
        input_encoded = pd.get_dummies(
    input_data,
    columns=["Fuel type","Model"],
    dtype=int
)
        # Align with dummy columns structure
        input_encoded = input_encoded.reindex(columns=dummy_columns, fill_value=0)

        # Scale numerical features
        num_cols = ['Year of manufacture','Mileage','Engine size']
        input_scaled = input_encoded.copy()
        input_scaled[num_cols] = scaler.transform(input_scaled[num_cols])

        # Predict price
        predicted_price = model.predict(input_scaled)[0]

        st.success(f"💰 Estimated Price: $ {predicted_price:,.2f}")
else:
    st.error("❌ Required model/scaler files not found. Please place .pkl files in the same folder as this script.")