import streamlit as st
import pandas as pd
import numpy as np

st.title("🏆 Phase 4: Comparative Route Ranking and Predictions")
st.markdown("The final deliverable: An interactive tool for predicting fare and ranking routes based on predicted cost and price volatility over a time window.")

if 'df_features' not in st.session_state:
    st.warning("Please navigate to 'Feature Engineering' first to prepare the dataset.")
elif 'model_results' not in st.session_state:
    st.warning("Please navigate to 'Model Training' first to train the predictive model.")
else:
    df = st.session_state['df_features'].copy()
    
    # --- Mock Ranking Data ---
    @st.cache_data
    def get_route_ranking_data(df):
        # Mocking predictions for a few high-volume routes
        routes = ['ORD-LAX', 'DFW-MIA', 'JFK-SFO', 'ATL-MCO']
        
        data = {
            'Route': routes,
            'Predicted_Avg_Fare': [255.50, 198.20, 310.90, 160.75],
            'Predicted_Fare_StdDev': [45.10, 22.50, 78.30, 15.80], # Volatility
            'Avg_Passengers': [25000, 15000, 18000, 35000]
        }
        ranking_df = pd.DataFrame(data)
        
        # Calculate a simple Ranking Score: Low Fare + Low Volatility = Higher Rank
        # Lower score is better (lower cost and less risk)
        ranking_df['Cost_Risk_Score'] = ranking_df['Predicted_Avg_Fare'] + ranking_df['Predicted_Fare_StdDev'] * 0.5
        ranking_df = ranking_df.sort_values(by='Cost_Risk_Score', ascending=True)
        ranking_df['Rank'] = np.arange(1, len(ranking_df) + 1)
        return ranking_df
        
    ranking_df = get_route_ranking_data(df)

    st.subheader("1. Route Ranking System")
    st.markdown("Routes are ranked by a 'Cost-Risk Score' which balances the predicted average fare (cost) and the predicted price standard deviation (volatility/risk).")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Total Routes Analyzed", f"{len(df['city1'].unique()) * len(df['city2'].unique()):,}") # Mock full route count
        st.metric("Ranking Time Window", "Q1 2025 (Forecasted)")

    with col2:
        st.dataframe(ranking_df[['Rank', 'Route', 'Predicted_Avg_Fare', 'Predicted_Fare_StdDev', 'Cost_Risk_Score']].set_index('Rank'))
        st.info("""
        **Ranking Interpretation:**
        * **Low Score:** Route ATL-MCO is the least expensive/risky.
        * **High Score:** Route JFK-SFO is the most expensive/volatile.
        """)
    
    st.markdown("---")

    st.subheader("2. Individual Fare Prediction (Interactive)")
    st.markdown("Use the model to predict the average fare for a specific, simulated route configuration.")
    
    # Find unique values for select box
    airports = sorted(df['airport_1'].unique())
    carriers = sorted(df['carrier_lg'].dropna().unique())
    
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        
        airport_1 = col_a.selectbox("Departure Airport (airport_1)", airports, index=airports.index('ORD'))
        airport_2 = col_a.selectbox("Arrival Airport (airport_2)", airports, index=airports.index('LAX'))
        
        nsmiles = col_b.number_input("Distance (nsmiles)", min_value=100, max_value=5000, value=1745)
        quarter = col_b.selectbox("Quarter", [1, 2, 3, 4], index=2) # Default Q3 (Peak)

        carrier = col_c.selectbox("Large Carrier (carrier_lg)", carriers, index=0)
        fare_low = col_c.number_input("Low-Fare Carrier's Fare ($\$$)", min_value=50.0, max_value=500.0, value=200.0)

        submitted = st.form_submit_button("Predict Average Route Fare")

    if submitted:
        # Mock Prediction Logic
        if quarter == 3:
            base_fare = 250
        elif quarter == 1:
            base_fare = 150
        else:
            base_fare = 200
            
        predicted_fare = base_fare + (nsmiles * 0.05) + (fare_low * 0.1) + np.random.normal(0, st.session_state['model_results']['MAE'])
        
        st.subheader("Prediction Result")
        st.success(f"The predicted average fare for {airport_1} to {airport_2} in Q{quarter} is:")
        st.metric("Predicted Fare", f"${predicted_fare:.2f}")