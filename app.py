import sys
import os
import streamlit as st
import pandas as pd
import random
from dotenv import load_dotenv
from utils.b2 import B2

# Add the utils directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# Load environment variables
load_dotenv()

# Set Backblaze connection
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 's3.us-east-005.backblazeb2.com'),
    key_id=os.getenv('B2_KEYID'),
    secret_key=os.getenv('B2_APPKEY')
)

def fetch_data():
    try:
        b2.set_bucket('AirBnB-Bucket')  # Set the bucket
        obj = b2.get_object('Airbnb Dataset_Long.csv')  # Use the EXACT file name
        return pd.read_csv(obj)
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None

# APPLICATION Title
st.title("Airbnb Data Viewer")

# Buyer and Seller buttons
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    st.header("Welcome to the Airbnb Explorer!")
    buyer = st.button("Buyer")
    seller = st.button("Seller")

    if buyer:
        st.session_state.page = "buyer"
    if seller:
        st.session_state.page = "seller"

# Fetch data from Backblaze and Preview to ensure its loaded
data = fetch_data()
if data is not None:
    st.write("Data loaded successfully.")
    st.dataframe(data.head())

# Placeholder for Buyer 
if st.session_state.page == "buyer":
    #Start Code here for Buyer side, replace the code below.
    st.write("Buyer window placeholder. Replace with  implementation.")

#Rough Draft Seller
elif st.session_state.page == "seller":
    st.header("Estimate Your Airbnb Listing Review Score")

    # Text inputs for seller
    neighborhood_overview = st.text_input("Neighborhood Overview")
    host_neighborhood = st.text_input("Host Neighborhood")
    property_type = st.text_input("Property Type")
    amenities = st.text_input("Included Amenities (comma separated)")
    price = st.text_input("Price")

    # Drop-down inputs for seller
    bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])
    beds = st.selectbox("Number of Beds", [1, 2, 3, 4, 5])

    if st.button("Generate Review Score"):
        # Generate a random score out of 5
        #This will be replaced with a proper predictive model
        review_score = round(random.uniform(1, 5), 2)
        st.success(f"Estimated Review Score: {review_score} out of 5")


    # Back button to go back to main page
    if st.button("Back"):
        st.session_state.page = "main"
