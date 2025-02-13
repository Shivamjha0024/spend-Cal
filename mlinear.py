import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

def load_model():
    # Dummy model training for demonstration
    model = LinearRegression()
    df = pd.read_excel(r"All_Data.xlsx")
    X = df[['Impression', 'Clicks', 'Leads']].values
    y = df['Spend'].values
    model.fit(X, y)
    return model

model = load_model()

def predict_spend(impressions, clicks, leads):
    input_data = np.array([[impressions, clicks, leads]])
    return model.predict(input_data)[0]

st.title("Spend Prediction Calculator")

# Load dataset for ranges
df = pd.read_excel(r"C:\Users\AGL IT\Downloads\MMM\All_Data.xlsx")

# Source selection
source = st.selectbox("Select Source", ["FB", "ADS"])

# Region selection
region = st.selectbox("Select Region", [
    'C1', 'C2', 'C3', 'C4', 'E1', 'E2', 'E3', 
    'N1', 'N2', 'N3', 'N4', 'S1', 'S2', 'S3', 
    'T1', 'T2', 'W1', 'W2', 'W3'
])

# Filter data based on selections
filtered_df = df[(df['Source'] == source) & (df['Region'] == region)]

# Feature inputs with manual input instead of slider
impressions = st.number_input(
    "Impressions",
    min_value=int(filtered_df['Impression'].min() if not filtered_df.empty else 1000),
    max_value=int(filtered_df['Impression'].max() if not filtered_df.empty else 100000),
    step=1000,
    value=int(filtered_df['Impression'].min() if not filtered_df.empty else 1000)
)

clicks = st.number_input(
    "Clicks",
    min_value=int(filtered_df['Clicks'].min() if not filtered_df.empty else 10),
    max_value=int(filtered_df['Clicks'].max() if not filtered_df.empty else 1000),
    step=10,
    value=int(filtered_df['Clicks'].min() if not filtered_df.empty else 10)
)

leads = st.number_input(
    "Leads",
    min_value=int(filtered_df['Leads'].min() if not filtered_df.empty else 1),
    max_value=int(filtered_df['Leads'].max() if not filtered_df.empty else 100),
    step=1,
    value=int(filtered_df['Leads'].min() if not filtered_df.empty else 1)
)

# Predict spend
predicted_spend = predict_spend(impressions, clicks, leads)

# Display result
st.markdown(f"## Predicted Spend: ₹{predicted_spend:,.2f}")

# UI styling
def style_ui():
    st.markdown(
        """
        <style>
            div.stSlider > div[data-baseweb="slider"] > div {
                background: #003366; border-radius: 10px;
            }
            div.stSlider > div[data-baseweb="slider"] > div > div > div {
                background: white;
            }
            .st-bb {border-radius: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

style_ui()
