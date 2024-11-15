# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from load_and_preprocess import load_model_components, preprocess_data

# Define prediction function
def predict(data):
    model_components = load_model_components('models/rf_model.joblib')
    inputs = pd.DataFrame(data, columns=model_components['input_cols'])
    X, inputs = preprocess_data(inputs, model_components)
    predictions = model_components['model'].predict(X)
    predict_proba = model_components['model'].predict_proba(X)
    return predictions[0], (predict_proba[0,1])

# Customize sidebar colour
#st.markdown("""
#    <style>
#        [data-testid="stSidebar"] {
#            background-color: #B0C1D9;
#        }
#    </style>
#""", unsafe_allow_html=True)

st.title('Rain prediction')
st.markdown('This application utilizes a Random Forest model to predict rainfall in Australia, leveraging 10 years of observational data')
url = "https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package"
st.write("Data source: Kaggle dataset ['Rain in Australia'](%s)" % url)
st.image('images/australia.jpg')


raw_df = pd.read_csv('data/weatherAUS.csv')
location_list = raw_df['Location'].dropna().unique()
WindGustDir_list = raw_df['WindGustDir'].dropna().unique()
WindDir9am_list = raw_df['WindDir9am'].dropna().unique()
WindDir3pm_list = raw_df['WindDir3pm'].dropna().unique()
st.markdown('1. Select the location in Australia You are interested in:')
Location = st.selectbox('Select location:', location_list)
st.markdown(f'Historical weather data for {Location} statistics:')
st.dataframe(raw_df[raw_df['Location']==Location].describe())

# Define lists for input widgets

st.markdown('2. Select the parameter values on the sidebar to predict the probability of rain for the next day.')

# Define sidebar with input widgets
with st.sidebar:
    st.markdown(f'Weather parameters for selected location:')
    RainToday = st.checkbox('It rained today')
    Rainfall = st.slider('Rainfall', 0, 400, 0)
    WindGustSpeed = st.slider('Wind Gust Speed', 0, 150, 75)
    WindGustDir = st.selectbox('Wind Gust Direction', WindGustDir_list)
    MinTemp = st.slider('Minimal Temperature', -15, 50, 20)
    MaxTemp = st.slider('Maximum Temperature', -15, 50, 20)
    Sunshine = st.slider('Sunshine', 0, 20, 10)
    Evaporation = st.slider('Evaporation', 0, 150, 75)
    WindDir9am = st.selectbox('Wind Direction at 9am', WindDir9am_list)
    WindDir3pm = st.selectbox('Wind Direction at 3pm', WindDir3pm_list)
    WindSpeed9am = st.slider('Wind Speed at 9am', 0, 150, 75)
    WindSpeed3pm = st.slider('Wind Speed at 3pm', 0, 150, 75)
    Humidity9am = st.slider('Humidity at 9am', 0, 100, 0)
    Humidity3pm = st.slider('Humidity at 3pm', 0, 100, 0)
    Pressure9am = st.slider('Pressure at 9am', 900, 1100, 1000)
    Pressure3pm = st.slider('Pressure at 3pm', 900, 1100, 1000)
    Cloud9am = st.slider('Cloud Cover at 9am', 0, 10, 5)
    Cloud3pm = st.slider('Cloud Cover at 3pm', 0, 10, 5)
    Temp9am = st.slider('Temperature at 9am', -15, 50, 20)
    Temp3pm = st.slider('Temperature at 3pm', -15, 50, 20)

# Predict button
st.markdown('3. Predict whether it will rain the next day (Yes / No).')

if st.button('Predict'):
    # Form a list with user data
    data = np.expand_dims(np.array([Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
                                    WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, 
                                    WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, 
                                    Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]), axis=0)
    
    # Call predict function
    pred, pred_proba = predict(data)
    st.write(f'Result: {pred}')
    st.write('Rain probability = {pred_proba: .4%}'.format(pred_proba=pred_proba))