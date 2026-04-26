import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('model_kn.pkl','rb'))

#scaling
scaler = StandardScaler()
st.title("house Price Prediction")

#input 
Square_Footage = st.number_input('Square Footage',min_value=1000 , max_value=5000,value=2500)
Num_Bedrooms = st.number_input('Number Of Bedrooms',min_value=1 , max_value=5,value=2)
Num_Bathrooms = st.number_input('Number Of Bathrooms',min_value=1 , max_value=3,value=1)
Year_Built = st.number_input('Year Built',min_value=1950 , max_value=2025,value=1990)
Lot_Size = st.number_input('Lot Size',min_value=0.1 , max_value=5.0,value=2.5)
Garage_Size = st.number_input('Garage Size',min_value=0 , max_value=3,value=1)
Neighborhood_Quality = st.number_input('Neighborhood Quality',min_value=1 , max_value=10,value=7)

# dataframe
input_features = pd.DataFrame({
    'Square_Footage':[Square_Footage],
    'Num_Bedrooms':[Num_Bedrooms],
    'Num_Bathrooms':[Num_Bathrooms],
    'Year_Built':[Year_Built],
    'Lot_Size':[Lot_Size],
    'Garage_Size':[Garage_Size],
    'Neighborhood_Quality':[Neighborhood_Quality]
})

if st.button('Predict'):
  predictions=model.predict(input_features)
  st.success(f"Price Prediction: ${predictions}")






