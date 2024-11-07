import os
import pickle
import streamlit as st
from dotenv import load_dotenv

# Import B2 from utils/b2.py
from utils.b2 import B2

# ------------------------------------------------------
#                      APP CONSTANTS
# ------------------------------------------------------
REMOTE_DATA = 'Airbnb Dataset_Final.xlsx'  # Update if your file name is different

# ------------------------------------------------------
#                        CONFIG
# ------------------------------------------------------
# Load environment variables
load_dotenv()

# Set Backblaze connection with new credentials
b2 = B2(
    endpoint=os.getenv('B2_ENDPOINT', 's3.us-east-005.backblazeb2.com'),  # Update if endpoint changes
    key_id=os.getenv('B2_KEYID', '005491ab29352f00000000003'),
    secret_key=os.getenv('B2_APPKEY', 'K005urBSkXoICdWzCf8QtT/CPxQCMy8')
)
