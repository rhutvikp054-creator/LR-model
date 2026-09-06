import streamlit as st
import joblib
import pandas as pd

# Load the trained model
loaded_lr = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction with Linear Regression')
st.write('Enter the advertising budgets for TV, Radio, and Newspaper to predict sales.')

# Input fields for features
tv = st.slider('TV Advertising Budget ($)', 0.0, 300.0, 150.0)
radio = st.slider('Radio Advertising Budget ($)', 0.0, 50.0, 25.0)
newspaper = st.slider('Newspaper Advertising Budget ($)', 0.0, 120.0, 30.0)

# Create a DataFrame for prediction
# The order of features must match the order used during training (TV, Radio, Newspaper)
input_data = pd.DataFrame([[tv, radio, newspaper]],
                            columns=['TV', 'Radio', 'Newspaper'])


# Make prediction
if st.button('Predict Sales'):
    prediction = loaded_lr.predict(input_data)
    st.success(f'Predicted Sales: {prediction[0]:.2f} units')
