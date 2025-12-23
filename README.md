## ✈️ Airfare-Price-Prediction ✈️
## &
## ✈️ Flight Route Analysis

**Machine Learning • Time Series Analysis • Geospatial Visualization**

This project analyzes U.S. domestic flight routes using data analytics, machine learning, and interactive geospatial visualization. It is implemented as a multi-page **Streamlit dashboard** covering data exploration, feature engineering, model training, forecasting, and interactive route mapping. 

---
## 📌 Project Overview

This project performs an end-to-end analysis of **U.S. domestic flight routes** using data analytics, machine learning, time series forecasting, and interactive geospatial visualization.

It is implemented as a **multi-page Streamlit dashboard**, enabling users to:

- Explore large-scale flight datasets  
- Engineer domain-specific features  
- Train and compare machine learning models  
- Perform time series analysis  
- Predict flight fares and passenger demand  
- Visually analyze routes on an interactive U.S. map  

---

## 🌟 Key Functional Modules

### 1️⃣ Exploratory Data Analysis (EDA)
- Dataset overview, schema inspection, and missing value analysis  
- Distribution analysis for:
  - Average fares  
  - Passenger volumes  
  - Route distances  
- Interactive filters and downloadable insights  

---

### 2️⃣ Feature Engineering
- Automated preprocessing pipeline:
  - Missing value handling  
  - Scaling and encoding  
- Domain-driven feature creation:
  - Seasonal indicators  
  - Distance buckets  
  - Demand classification  
  - Fare normalization  
- Modular and reusable utilities implemented in `utils/`  

---

### 3️⃣ Model Training & Evaluation
- Trained and evaluated multiple machine learning models:
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting  
- Performance comparison using:
  - RMSE  
  - MAE  
  - R² Score  
- Error analysis and residual visualizations for better interpretability  

---

### 4️⃣ Time Series Analysis
- Monthly and yearly trend analysis  
- Seasonality detection and decomposition  
- Forecasting using:
  - ARIMA  
  - Prophet (where applicable)  
- Trend, seasonality, and residual decomposition plots  

---

### 5️⃣ Prediction & Route Ranking
- Predicts:
  - Average flight fares  
  - Passenger volumes  
- Ranks routes based on:
  - Highest predicted fare  
  - Highest demand  
  - Lowest predicted cost  
- Interactive controls with CSV export functionality  

---

### 6️⃣ Geospatial Route Visualization
- Interactive **Folium-based U.S. route map**  
- Visual encoding:
  - Line thickness → Passenger volume  
  - Line color → Average fare  
- City-to-city route popups with detailed metrics  
- Sidebar controls:
  - Fare filters  
  - Passenger thresholds  
  - Route count  
  - Opacity and color scheme  
- Supporting visualizations:
  - Fare distribution  
  - Passenger distribution  
  - Fare vs. passenger scatter plots  

---

### 📁 Project Structure
```bash 
Flight_Route_Analysis/
│
├── app.py                      # Main Streamlit app entry point
├── README.md
├── requirements.txt
│
├── data/
│   └── flight_data.csv         # Dataset (~63 MB)
│
├── pages/                      # Streamlit multipage files
│   ├── 1_dataset_eda.py
│   ├── 2_feature_engineering.py
│   ├── 3_model_training.py
│   ├── 4_time_series_analysis.py
│   ├── 5_prediction_ranking.py
│   └── 6_route_visualization.py
│
└── utils/                      # Reusable functions
    ├── preprocessing.py
    ├── feature_engineering.py
    └── modeling.py

```
### 🧠 Technologies Used

* Python 3.x
* Streamlit
* Pandas / NumPy
* Scikit-learn
* Folium & streamlit-folium
* Matplotlib & Seaborn
* Statsmodels / Prophet (if used for forecasting)

### 📊 Dataset Information

* The dataset includes U.S. domestic flight routes with:

  * Origin & destination cities
  * Geocoded coordinates
  * Monthly passenger counts
  * Average fares
  * Route distance
  * Time period indicators
  * Large file size: ~63 MB, loaded efficiently with caching.

### 🔮 Future Enhancements

* Add real-time API for live airfare updates.
* Integrate LSTM or Prophet models for more accurate forecasting.
* Add clustering to identify route demand groups.
* Build airline-specific dashboards.
* Add performance benchmarking for models.

## 👤 Author

**Yashwanth Devagudi Veeravenkata**

---

## 📜 License

This project is released under the **MIT License** and is open-source.
