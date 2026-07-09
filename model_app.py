import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    # Resolve relative to this script so it works regardless of cwd
    model_path = Path(__file__).parent / "xgb_car_price_model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure the model is trained and saved.")
    st.stop()

# Title and description
st.title("🚗 Car Price Predictor")
st.markdown("Predict the selling price of your car based on its features")
st.divider()

# Create two columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Car Details")
    
    # Numeric inputs
    km_driven = st.slider(
        "Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=50000,
        step=10000,
        help="Total kilometers driven by the car"
    )
    
    mileage = st.slider(
        "Mileage (km/l)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        help="Fuel efficiency in kilometers per liter"
    )
    
    age = st.slider(
        "Age (Years)",
        min_value=0,
        max_value=25,
        value=5,
        step=1,
        help="Age of the car in years"
    )

with col2:
    st.subheader("⛽ Fuel Type")
    
    fuel_type = st.radio(
        "Select Fuel Type",
        options=["Petrol", "Diesel", "Electric"],
        horizontal=False,
        help="Choose the primary fuel type of the vehicle"
    )
    
    # Create binary features for fuel type
    petrol = 1 if fuel_type == "Petrol" else 0
    diesel = 1 if fuel_type == "Diesel" else 0
    electric = 1 if fuel_type == "Electric" else 0

st.divider()

# Prediction button
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):
    # Prepare features in the same order as training
    features = pd.DataFrame({
        'km_driven': [km_driven],
        'mileage': [mileage],
        'age': [age],
        'Petrol': [petrol],
        'Diesel': [diesel],
        'Electric': [electric]
    })
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    st.divider()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Price",
            value=f"₹ {prediction:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Fuel Type",
            value=fuel_type
        )
    
    with col3:
        st.metric(
            label="Car Age",
            value=f"{age} years"
        )
    
    # Display input summary
    st.subheader("📋 Input Summary")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.write(f"**Kilometers Driven:** {km_driven:,} km")
        st.write(f"**Mileage:** {mileage} km/l")
    
    with summary_col2:
        st.write(f"**Age:** {age} years")
        st.write(f"**Fuel Type:** {fuel_type}")
    
    # Visual representation
    st.subheader("💰 Price Range Estimate")
    price_lower = prediction * 0.85
    price_upper = prediction * 1.15
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📉 Lower Estimate\n**₹ {price_lower:,.0f}**")
    with col2:
        st.success(f"✓ Predicted Price\n**₹ {prediction:,.0f}**")
    with col3:
        st.warning(f"📈 Upper Estimate\n**₹ {price_upper:,.0f}**")

st.divider()
st.caption("💡 This prediction is based on an XGBoost machine learning model trained on historical car data.")