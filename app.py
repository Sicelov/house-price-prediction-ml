import streamlit as st
import pickle
import pandas as pd

# load model and encoders
model = pickle.load(open('housing_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

st.title('Housing Price Prediction')

# Input fields for user data
area = st.number_input('Area (in sq ft)', min_value=500, max_value=20000)
bedrooms = st.slider('Bedrooms', 1, 6)
bathrooms = st.slider('Bathrooms', 1, 4)
stories = st.slider('Stories', 1, 4)
mainroad = st.selectbox('Main Road', ['Yes', 'No'])
guestroom = st.selectbox('Guest Room', ['Yes', 'No'])
basement = st.selectbox('Basement', ['Yes', 'No'])
hotwaterheating = st.selectbox('Hot Water Heating', ['Yes', 'No'])
airconditioning = st.selectbox('Air Conditioning', ['Yes', 'No'])
parking = st.slider('Parking', 0, 3)
prefarea = st.selectbox('Preferred Area', ['Yes', 'No'])
furnishingstatus = st.selectbox(
    'Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])


# Predict button
if st.button('Predict Price'):
    # Prepare input data
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    # Encode categorical variables
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted House Price: ${prediction:,.2f}')
