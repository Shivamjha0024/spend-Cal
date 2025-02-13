# Spend Prediction Web App

This repository contains a Spend Prediction web application built using Flask and Streamlit.

## Features
- Train a Linear Regression model on a dataset.
- Predict advertising spend based on impressions, clicks, and leads.
- Web API using Flask for model training and prediction.
- Interactive UI using Streamlit for a user-friendly experience.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Files Overview

- **mlinear.py**: Streamlit UI for interactive spend prediction.
- **api_mmm_v1.py**: Flask-based API for predicting spend.
- **All_Data.xlsx**: Dataset containing historical ad performance data.

## Running the Application

### 1. Start the Flask API
```bash
python api_mmm_v1.py
```

### 2. Run the Streamlit UI
```bash
streamlit run mlinear.py
```

## API Endpoints

### 1. Predict Spend
**Endpoint:** `POST /predict`
- Provide JSON data with `source, region, impressions, clicks, leads`.

#### Example request:
```json
{
  "source": "FB",
  "region": "C1",
  "impressions": 5000,
  "clicks": 200,
  "leads": 10
}
```

#### Example response:
```json
{
  "predicted_spend": 250.75
}
```

## UI Usage
1. Upload an Excel file (`All_Data.xlsx`) with required columns.
2. Select `Source` and `Region`.
3. Enter `Impressions`, `Clicks`, and `Leads`.
4. View the predicted spend.

## Dependencies
- Flask
- NumPy
- Pandas
- Streamlit
- Scikit-learn

