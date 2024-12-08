import os
import pandas as pd
import streamlit as st
import pydeck as pdk
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from utils.modeling_sentiment import encode_property_type, load_model
import sys
from utils.b2 import B2
from dotenv import load_dotenv
from io import BytesIO

# Add the utils directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# Load environment variables
load_dotenv()

# Load Backblaze from Streamlit Secrets
endpoint = os.getenv("B2_ENDPOINT")  
key_id = os.getenv("B2_KEYID")
app_key = os.getenv("B2_APPKEY")
bucket_name = os.getenv("B2_BUCKETNAME")

# Set up Backblaze connection
b2 = B2(
    endpoint=endpoint,
    key_id=key_id,
    secret_key=app_key
)

@st.cache_data 
def fetch_data():
    try:
        b2.set_bucket(bucket_name)  
        obj = b2.get_object('Final_PROJ.xlsx')  # Use Exact Name of File

        # Convert the StreamingBody object to a BytesIO object
        file_content = obj.read()  # Read the content of the StreamingBody
        return pd.read_excel(BytesIO(file_content))  # Use BytesIO to create a file-like object
    except Exception as e:
        st.error(f"Error fetching data from Backblaze: {e}")
        return None


def get_sentiment_score(text, analyzer):
    """Utility function to get sentiment score using SentimentIntensityAnalyzer."""
    if text:
        sentiment = analyzer.polarity_scores(text)
        return sentiment['compound']
    return None  # Default sentiment score if text is missing


# Load trained model and scaler from pickle file
try:
    model, scaler, expected_features = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please add the trained model.pickle.")
    st.stop()

# Streamlit UI
st.title("Airbnb Data Viewer")

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = "Main"

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Main application function
def main():
    # Set the page config for a wide layout
    st.set_page_config(page_title="Airbnb Data Viewer", layout="wide", initial_sidebar_state="expanded")

    # CSS styling
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f9;
            }
            .title {
                font-size: 3rem;
                font-weight: bold;
                color: #F0000B;
                text-align: center;
                margin-top: 40px;
            }
            .sub-title {
                font-size: 2rem;
                color: #333;
            }
            .markdown-text {
                font-size: 1.1rem;
                line-height: 1.6;
                color: #555;
            }
            .stButton>button {
                background-color: #007BFF;
                color: white;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 20px;
            }
            .stButton>button:hover {
                background-color: #006FCE;
            }
            .sidebar .sidebar-content {
                background-color: #2F4F4F;
                color: white;
            }
            .stSidebar .sidebar-content .element-container {
                color: white;
            }
            .footer {
                background-color: #007BFF;
                padding: 10px;
                color: white;
                text-align: center;
                font-size: 14px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title Section with custom color
    st.markdown('<h1 class="title">Airbnb Explorer</h1>', unsafe_allow_html=True)

    # Navigation
    navigation = st.sidebar.selectbox("Navigate", ["Main", "Buyer Page", "Seller Page"])

    # Fetch data
    data = fetch_data()

    # Main Page Content with Tabs
    if navigation == "Main":
        st.header("Welcome to the Airbnb Data Explorer")

        # Tab system
        tab = st.selectbox("Select a section", ["Introduction", "Goals and Approach", "Data Preview"])

        if tab == "Introduction":
            st.markdown('<p class="sub-title">Introduction</p>', unsafe_allow_html=True)
            st.markdown("""
            <p class="markdown-text">Using this data, we aim to uncover insights into what makes an Airbnb listing successful.</p>
            """, unsafe_allow_html=True)

        elif tab == "Goals and Approach":
            st.markdown('<p class="sub-title">Our Goals and Approach</p>', unsafe_allow_html=True)
            st.markdown("""
            <p class="markdown-text">
            Our Goals:
            - Understand the factors that lead to higher ratings and greater foot traffic.
            - Explore the influence of location and neighborhoods on reviews and ratings.
            - Identify specific amenities that contribute to higher reviews.
            - Evaluate if pricing strategies impact success.

            Our Approach:
            - Compare locations and neighborhoods to identify "hotspots" that are favored by users.
            - Assess the role of amenities in improving customer satisfaction.
            - Analyze price ranges to find the "sweet spot" that attracts the most guests.
            </p>
            """, unsafe_allow_html=True)

        elif tab == "Data Preview":
            if data is not None and 'id' in data.columns:
                # Convert 'id' to integer, then keep only the first five digits
                data['id'] = data['id'].apply(lambda x: str(int(float(x)))[:5])

            # Display data on the main page
            if data is not None:
                st.write("Data loaded successfully.")
                st.dataframe(data.head())
            else:
                st.write("Failed to load data.")

    # Buyer Page
    elif navigation == "Buyer Page" and data is not None:
        st.header("Buyer Page")

        # Inputs with better styling
        rating_input = st.number_input("Minimum Review Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
        price_input = st.number_input("Maximum Price ($)", min_value=0, value=500)

        unique_property_types = (
            ["Any"] + sorted(data['property_type'].dropna().unique().tolist()) 
            if 'property_type' in data.columns else ["Any"]
        )
        selected_property_type = st.selectbox("Property Type", options=unique_property_types)

        unique_bedrooms = (
            sorted(data['bedrooms'].dropna().unique()) 
            if 'bedrooms' in data.columns else []
        )
        selected_bedrooms = st.selectbox("Number of Bedrooms", options=unique_bedrooms)

        search_button = st.button("Search")

        if search_button:
            filtered_data = data.copy()

            # Filter by rating
            if 'review_scores_rating' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['review_scores_rating'] >= rating_input]

            # Filter by property type
            if selected_property_type != "Any" and 'property_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['property_type'] == selected_property_type]

            # Filter by price
            if 'price' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['price'] <= price_input]

            # Filter by bedrooms
            if 'bedrooms' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['bedrooms'] == selected_bedrooms]

            # Display filtered data
            if len(filtered_data) > 0:
                st.write(f"Found {len(filtered_data)} properties based on your search criteria.")
                st.dataframe(filtered_data[['review_scores_rating', 'name', 'listing_url', 'price', 'bedrooms']])

                # Render map
                if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
                    filtered_data = filtered_data.dropna(subset=['latitude', 'longitude'])
                    deck = pdk.Deck(
                        map_style='mapbox://styles/mapbox/streets-v11',
                        initial_view_state=pdk.ViewState(
                            latitude=filtered_data['latitude'].mean(),
                            longitude=filtered_data['longitude'].mean(),
                            zoom=10,
                            pitch=50,
                        ),
                        layers=[pdk.Layer(
                            'ScatterplotLayer',
                            data=filtered_data,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=200,
                            pickable=True
                        )],
                        tooltip={
                            "html": "<b>Listing Name:</b> {name}<br/><b>Price:</b> {price}<br/><b>Review Score:</b> {review_scores_rating}",
                            "style": {"backgroundColor": "steelblue", "color": "white"}
                        }
                    )
                    st.pydeck_chart(deck)
                else:
                    st.error("Latitude and longitude columns are missing or invalid.")
            else:
                st.write("No properties match your search criteria.")

    # Seller Page
    elif navigation == "Seller Page":
        st.sidebar.title("Seller's Property Details")

        # User inputs for prediction
        accommodates = st.sidebar.number_input("Accommodates", min_value=1, step=1)
        bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.0, step=0.1)
        bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, step=1)
        price = st.sidebar.number_input("Price ($)", min_value=0, step=1)
        property_type = st.sidebar.selectbox("Property Type", options=["House", "Apartment", "Condo", "Studio"])

        predict_button = st.sidebar.button("Predict Review Score")

        if predict_button:
            # Get sentiment analysis score
            analyzer = SentimentIntensityAnalyzer()
            sentiment_score = get_sentiment_score("Example review text", analyzer)

            # Encoding property type
            property_type_encoded = encode_property_type(property_type)

            # Prepare features for prediction
            features = [accommodates, bathrooms, bedrooms, price, property_type_encoded, sentiment_score]
            scaled_features = scaler.transform([features])

            # Make prediction
            predicted_score = model.predict(scaled_features)[0]
            st.write(f"Predicted Review Score: {predicted_score:.2f}")

if __name__ == "__main__":
    main()
