from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Load model
def load_model():
    # Dummy model training for demonstration
    model = LinearRegression()
    df = pd.read_excel(r"C:\Users\AGL IT\Downloads\MMM\All_Data.xlsx")
    X = df[['Impression', 'Clicks', 'Leads']].values
    y = df['Spend'].values
    model.fit(X, y)
    return model

model = load_model()

def predict_spend(impressions, clicks, leads):
    input_data = np.array([[impressions, clicks, leads]])
    return model.predict(input_data)[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Extract inputs for source, region, and features
        source = data['source']
        region = data['region']
        impressions = data['impressions']
        clicks = data['clicks']
        leads = data['leads']

        # Load dataset for filtering
        df = pd.read_excel(r"All_Data.xlsx")

        # Filter data based on source and region
        filtered_df = df[(df['Source'] == source) & (df['Region'] == region)]

        # Adjust feature input ranges based on filtered data
        min_impressions = int(filtered_df['Impression'].min()) if not filtered_df.empty else 1000
        max_impressions = int(filtered_df['Impression'].max()) if not filtered_df.empty else 100000
        min_clicks = int(filtered_df['Clicks'].min()) if not filtered_df.empty else 10
        max_clicks = int(filtered_df['Clicks'].max()) if not filtered_df.empty else 1000
        min_leads = int(filtered_df['Leads'].min()) if not filtered_df.empty else 1
        max_leads = int(filtered_df['Leads'].max()) if not filtered_df.empty else 100

        # Ensure input values are within valid ranges
        impressions = max(min_impressions, min(impressions, max_impressions))
        clicks = max(min_clicks, min(clicks, max_clicks))
        leads = max(min_leads, min(leads, max_leads))

        # Predict spend
        predicted_spend = predict_spend(impressions, clicks, leads)

        # Return response
        return jsonify({
            "predicted_spend": round(predicted_spend, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
