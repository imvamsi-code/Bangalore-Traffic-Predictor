import streamlit as st
import pandas as pd
import datetime
import joblib

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Traffic Predictor", page_icon="🚦", layout="centered")
st.title("🚦 Bangalore Traffic Predictor")
st.write("Predict live traffic volume and congestion levels for major intersections.")

# --- 2. LOAD THE FROZEN MODELS ---
@st.cache_resource  # This keeps the models in memory so the app runs lightning fast!
def load_models():
    reg = joblib.load('Traffic-Predictor-App/xgboost_traffic_regressor.pkl')
    clf = joblib.load('Traffic-Predictor-App/xgboost_traffic_classifier.pkl')
    cols = joblib.load('Traffic-Predictor-App/model_columns.pkl')
    return reg, clf, cols

try:
    reg_model, clf_model, model_columns = load_models()
except Exception as e:
  st.error(f"⚠️ System Error: {e}")
  st.stop()

# --- 3. USER INPUTS (THE DASHBOARD) ---
col1, col2 = st.columns(2)

with col1:
    selected_road = st.selectbox("📍 Intersection Name:", 
                                ("Silk Board Junction", "ITPL Main Road", "Hosur Road", "Hebbal Flyover", "M.G. Road"))
    selected_area = st.selectbox("🗺️ Area Name:", 
                                ("Koramangala", "Whitefield", "Electronic City", "Indiranagar"))
    selected_weather = st.selectbox("☀️ Weather:", 
                                   ("Clear", "Rain", "Fog", "Overcast", "Windy"))

with col2:
    selected_date = st.date_input("📅 Date:", datetime.date(2026, 6, 29))
    selected_time = st.time_input("⏰ Time:", datetime.time(18, 30))
    pedestrians = st.slider("🚶 Pedestrian Count:", 0, 1000, 250)
    roadwork = st.radio("🚧 Active Roadwork?", ("No", "Yes"))

# --- 4. PREDICTION BUTTON ---
if st.button("🚀 Predict Traffic Now", use_container_width=True):
    
    # Create a blank row of 0s matching the exact training columns
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Insert Time Features
    input_df['Month'] = selected_date.month
    input_df['Day'] = selected_date.day
    input_df['DayOfWeek'] = selected_date.weekday()
    input_df['Hour'] = selected_time.hour
    
    # Insert Numerical Features
    input_df['Pedestrian and Cyclist Count'] = pedestrians
    input_df['Roadwork and Construction Activity'] = 1 if roadwork == 'Yes' else 0
    
    # Insert One-Hot Encoded Features
    weather_col = f"Weather Conditions_{selected_weather}"
    area_col = f"Area Name_{selected_area}"
    road_col = f"Road/Intersection Name_{selected_road}"
    
    # Safely flip the switches to 1 if they exist
    if weather_col in input_df.columns: input_df[weather_col] = 1
    if area_col in input_df.columns: input_df[area_col] = 1
    if road_col in input_df.columns: input_df[road_col] = 1
    
    # Generate Predictions
    predicted_cars = reg_model.predict(input_df)[0]
    category_num = clf_model.predict(input_df)[0]
    
    category_map = {0: '🟢 Low', 1: '🟡 Normal', 2: '🔴 Heavy'}
    predicted_category = category_map.get(category_num, "Unknown")
    
    # Display Results beautifully
    st.divider()
    st.subheader("📊 Prediction Results")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Estimated Motorized Vehicles", f"{int(predicted_cars)} units")
    res_col2.metric("Expected Congestion", predicted_category)
