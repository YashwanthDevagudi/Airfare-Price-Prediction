# Airfare-Price-Prediction

✈️ Flight Route Analysis & Forecasting Dashboard

Machine Learning • Time Series Analysis • Geospatial Visualization 

📌 Project Overview

This project performs an end-to-end analysis of U.S. domestic flight routes using data analytics, machine learning, time series forecasting, and interactive geospatial visualization.

It is implemented as a multi-page Streamlit dashboard, enabling users to:

Explore large-scale flight datasets

Engineer domain-specific features

Train and compare ML models

Perform time series analysis

Predict fares and passenger demand

Visually analyze routes on an interactive U.S. map

🌟 Key Functional Modules
1️⃣ Exploratory Data Analysis (EDA)

Dataset overview, schema, and missing value analysis

Distribution analysis for:

Average fares

Passenger volumes

Route distances

Interactive filters and downloadable insights

2️⃣ Feature Engineering

Automated preprocessing pipeline:

Missing value handling

Scaling and encoding

Domain-driven features:

Seasonal indicators

Distance buckets

Demand classification

Fare normalization

Modular and reusable utilities implemented in utils/

3️⃣ Model Training & Evaluation

Trained and evaluated multiple ML models:

Linear Regression

Random Forest Regressor

Gradient Boosting

Performance comparison using:

RMSE

MAE

R² Score

Error and residual visualizations for interpretability

4️⃣ Time Series Analysis

Monthly and yearly trend analysis

Seasonality detection and decomposition

Forecasting using:

ARIMA

Prophet (where applicable)

Trend, seasonality, and residual decomposition plots

5️⃣ Prediction & Route Ranking

Predict:

Average flight fares

Passenger volumes

Rank routes based on:

Highest predicted fare

Highest demand

Lowest predicted cost

Interactive controls with CSV export functionality

6️⃣ Geospatial Route Visualization

Interactive Folium-based U.S. route map

Visual encoding:

Line thickness → Passenger volume

Line color → Average fare

City-to-city route popups with detailed metrics

Sidebar controls:

Fare filters

Passenger thresholds

Route count

Opacity and color scheme

Supporting plots:

Fare distribution

Passenger distribution

Fare vs passenger scatter plots
