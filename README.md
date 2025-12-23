# ✈️ Airfare-Price-Prediction ✈️

# ✈️ Flight Route Analysis & Forecasting Dashboard

**Machine Learning • Time Series Analysis • Geospatial Visualization 
---

## 📌 Project Overview

This project performs an end-to-end analysis of **U.S. domestic flight routes** using data analytics, machine learning, time series forecasting, and interactive geospatial visualization.

It is implemented as a **multi-page Streamlit dashboard**, allowing users to explore flight data, engineer features, train predictive models, analyze trends, and visualize routes on an interactive U.S. map.

---

## 🌟 Key Features

### 1️⃣ Exploratory Data Analysis (EDA)
- Dataset overview, schema inspection, and missing value analysis  
- Distribution analysis of fares, passenger volumes, and route distances  
- Interactive filters and exportable insights  

---

### 2️⃣ Feature Engineering
- Automated preprocessing pipeline  
- Handling missing values, scaling, and encoding  
- Domain-driven feature creation:
  - Seasonal indicators  
  - Distance buckets  
  - Demand categories  
  - Fare normalization  
- Reusable and modular utilities inside the `utils/` directory  

---

### 3️⃣ Model Training & Evaluation
- Trained multiple machine learning models:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting  
- Model evaluation using:
  - RMSE  
  - MAE  
  - R² Score  
- Visualization of prediction errors and residuals  

---

### 4️⃣ Time Series Analysis
- Monthly and yearly trend analysis  
- Seasonality detection and decomposition  
- Forecasting using ARIMA and Prophet (where applicable)  
- Trend, seasonality, and residual decomposition plots  

---

### 5️⃣ Prediction & Route Ranking
- Predicts average flight fares and passenger volumes  
- Ranks routes by:
  - Highest predicted fare  
  - Highest demand  
  - Lowest predicted cost  
- Interactive controls with CSV download support  

---

### 6️⃣ Geospatial Route Visualization
- Interactive U.S. flight route map built using Folium  
- Visual encodings:
  - Route thickness represents passenger volume  
  - Route color represents average fare  
- City-to-city route popups with detailed metrics  
- Sidebar controls for fare filters, passenger thresholds, opacity, and route count  
- Supporting visualizations using Matplotlib and Seaborn  

---

## 📁 Project Structure

Flight_Route_Analysis/
│
├── app.py                      # Main Streamlit application
├── README.md                   # Project documentation (written by me)
├── requirements.txt
│
├── data/
│   └── flight_data.csv         # Large dataset (~63 MB)
│
├── pages/                      # Streamlit multipage modules
│   ├── 1_dataset_eda.py
│   ├── 2_feature_engineering.py
│   ├── 3_model_training.py
│   ├── 4_time_series_analysis.py
│   ├── 5_prediction_ranking.py
│   └── 6_route_visualization.py
│
└── utils/                      # Reusable utility functions
    ├── preprocessing.py
    ├── feature_engineering.py
    └── modeling.py
