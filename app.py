import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
with open('housing_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title('🏠 House Price Prediction')

# Input form
area = st.number_input('Area (sq ft)', min_value=500, max_value=20000, value=7500)
bedrooms = st.slider('Bedrooms', 1, 6, 3)
bathrooms = st.slider('Bathrooms', 1, 4, 2)
stories = st.slider('Stories', 1, 4, 2)
mainroad = st.selectbox('Main Road', ['yes', 'no'], index=0)
guestroom = st.selectbox('Guest Room', ['yes', 'no'], index=0)
basement = st.selectbox('Basement', ['yes', 'no'], index=0)
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'], index=0)
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'], index=1)
parking = st.slider('Parking Spaces', 0, 3, 1)
prefarea = st.selectbox('Preferred Area', ['yes', 'no'], index=1)
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'], index=0)

    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        predict_button = st.form_submit_button('Predict Price')
    
    with col2:
        reset_button = st.form_submit_button('Reset Inputs')

    # If reset button is clicked, Streamlit will rerun and reset all values to defaults
    if reset_button:
        st.experimental_rerun()

if st.button('Predict Price'):
    try:
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
        
        # Encode categorical variables with error handling
        for col in label_encoders:
            if col in input_data.columns:
                # Handle unseen labels by defaulting to first category
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0]
                )
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted House Price: ₹{prediction:,.2f}')
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.write("Debug Info:")
        st.json(input_data.to_dict())
