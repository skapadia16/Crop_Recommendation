import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Title and description
st.title("🌱 Crop Recommendation System")
st.markdown("Enter the soil and environmental parameters to get the recommended crop for cultivation.")

# Load data and train model (cached for performance)
@st.cache_resource
def load_and_train_model():
    # Load the dataset
    data = pd.read_csv("Crop_recommendation.csv")

    # Prepare features and target
    X = data.drop("label", axis=1)
    y = data["label"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

# Load the model
model, feature_names = load_and_train_model()

# Create input form
st.subheader("Enter Soil & Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42, step=1)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43, step=1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.8, step=0.1)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.01)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.9, step=0.1)

# Predict button
if st.button("Get Crop Recommendation", type="primary"):
    # Prepare input data as DataFrame with proper column names
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities (for debugging)
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Display result
    st.success(f"## Recommended Crop: **{prediction.upper()}**")

    # Show crop image (using placeholder or emoji based on crop name)
    # Dictionary mapping crop names to emojis (as placeholders for images)
    crop_emojis = {
        'rice': '🌾',
        'maize': '🌽',
        'chickpea': '🥜',
        'kidneybeans': '🫘',
        'pigeonpeas': '🫘',
        'mothbeans': '🫘',
        'mungbean': '🫘',
        'blackgram': '🫘',
        'lentil': '🫘',
        'pomegranate': '🍇',
        'banana': '🍌',
        'mango': '🥭',
        'grapes': '🍇',
        'watermelon': '🍉',
        'muskmelon': '🍈',
        'apple': '🍎',
        'orange': '🍊',
        'papaya': '🥭',
        'coconut': '🥥',
        'cotton': '🧵',
        'jute': '🪢',
        'coffee': '☕'
    }

    # Get emoji for the predicted crop
    emoji = crop_emojis.get(prediction.lower(), '🌱')

    # Display the emoji large
    st.markdown(f"<div style='text-align: center; font-size: 100px;'>{emoji}</div>", unsafe_allow_html=True)

    # Also show the crop name as caption
    st.markdown(f"<div style='text-align: center; font-size: 24px; margin-top: -20px;'>{prediction.title()}</div>", unsafe_allow_html=True)

    # Show top 3 predictions with probabilities (for debugging/transparency)
    st.subheader("Top 3 Predictions:")
    # Get indices of top 3 probabilities
    top_3_idx = np.argsort(probabilities)[-3:][::-1]
    for idx in top_3_idx:
        crop = classes[idx]
        prob = probabilities[idx]
        emoji = crop_emojis.get(crop.lower(), '🌱')
        st.write(f"{emoji} {crop.title()}: {prob:.1%}")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info(
        """
        This crop recommendation system uses a Random Forest classifier
        trained on agricultural data to suggest the most suitable crop
        based on soil nutrients and environmental conditions.

        **Parameters:**
        - N: Nitrogen content in soil
        - P: Phosphorus content in soil
        - K: Potassium content in soil
        - temperature: Temperature in Celsius
        - humidity: Relative humidity in percentage
        - ph: pH value of the soil
        - rainfall: Rainfall in mm
        """
    )

    st.header("Model Info")
    st.text("Algorithm: Random Forest")
    st.text("Estimators: 100")
    st.text("Training Accuracy: ~99.3%")

    # Add a debug section
    st.header("Debug Info")
    if st.checkbox("Show input values"):
        st.write("Input values:")
        st.write(f"N: {N}, P: {P}, K: {K}")
        st.write(f"Temperature: {temperature}°C")
        st.write(f"Humidity: {humidity}%")
        st.write(f"pH: {ph}")
        st.write(f"Rainfall: {rainfall} mm")