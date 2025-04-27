#importing all the necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Loading the dataset
house_data=pd.read_csv('Mumbai_NEW.csv')

# Configuring the web page
st.set_page_config(
    page_title="House Price Prediction", page_icon=":office_building:", layout="wide", initial_sidebar_state="expanded"
)

st.title("Housing Price Prediction App")

st.markdown("""
<style>
    .main{
        background-color: #f5f5f5;
        padding : 20px:
    }
    h1{
      text-align : center;
      color : #003366;
    }
    
</style>
""", unsafe_allow_html=True)

# Visualizing the current trends
st.subheader("Visualizing Housing Data Trends")

#Bar Chart
st.header("Average price by location")
location_price = house_data.groupby('Location')['Price'].mean().sort_values(ascending=False).head(50)
st.bar_chart(location_price)

#Pie Chart
st.header("New vs Resale Property Distribution")
property_counts = house_data['New/Resale'].value_counts()
st.write(property_counts)

fig = px.pie(
    names=property_counts.index, values=property_counts.values, title="New vs Resale Properties",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig)

#Map 
st.header("Properties on map")
map_data = pd.DataFrame({
    'lat': [19.218330],
    'lon': [72.978088]

})
st.map(map_data)

# Loading the pickle file and making the predictions

locations = sorted(house_data['Location'].unique())
st.subheader("Scroll up and explore the prediction section!")
st.subheader("Predict the future value of your dream home with accuracy")
model = pickle.load(open('house_price_model.pkl', 'rb'))
location_encoder = pickle.load(open('location_encoder.pkl', 'rb'))
st.write("Use the sideBar to enter the property details  : ")

with st.sidebar:
    st.header("Enter House Details : ")
    Area = st.number_input('Enter Area : ', min_value=00)
    Location = st.selectbox("Select Location", locations)
    BHK = st.number_input('Enter BHK : ', min_value=1, max_value=7)
    New_Resale = st.selectbox("Want a new property or Resale?",['New', 'Resale'])
    Lift = st.selectbox("Lift Available?", ['Yes', 'No'])
    Parking = st.selectbox("Parking Available?", ['Yes','No'])
    Security = st.selectbox("24x7 Security Available",['Yes','No'])

    new_resale_encoded = 1 if New_Resale == 'New' else 0
    parking_encoded = 1 if Parking == 'Yes' else 0
    lift_encoded = 1 if Lift == 'Yes' else 0
    security_encoded = 1 if Security == 'Yes' else 0

    submit = st.button("Predict Price")

if submit:
    encoded_location = location_encoder.transform([Location])[0]
    input_data = np.array([[Area,encoded_location,BHK,new_resale_encoded,lift_encoded,parking_encoded,security_encoded]])

    with st.spinner("Predicting your dream house price...."):
        prediction = model.predict(input_data)

    st.success(f"The approximate predicted house price is : INR {prediction[0]:,.2f}")

print()


    




















    
    

    
