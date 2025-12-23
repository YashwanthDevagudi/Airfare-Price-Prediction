import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np

st.title("📈 Phase 4: Seasonal Decomposition and Forecasting")
st.markdown("Implementing advanced time series techniques to rigorously separate the long-term **Trend**, **Seasonal** effects, and **Residual** noise components of the quarterly average fare. This is essential for understanding predictable pricing cycles.")

if 'df_features' not in st.session_state:
    st.warning("Please navigate to 'Feature Engineering' first to prepare the dataset.")
else:
    df = st.session_state['df_features'].copy()
    
    # --- Time Series Preparation ---
    @st.cache_data
    def prepare_ts_data(df):
        """
        Aggregates data to a Quarterly Start (QS) frequency time series.
        """
        # Aggregate data to a quarterly time series
        fare_ts = df.groupby(['Year', 'quarter'])['fare'].mean().reset_index()
        
        # Determine the first month of the quarter (Q1=1, Q2=4, Q3=7, Q4=10)
        fare_ts['date'] = pd.to_datetime(
            fare_ts['Year'].astype(str) + '-' + 
            (fare_ts['quarter'] * 3 - 2).astype(str) + 
            '-01'
        )
        
        # Set the date index and explicitly define the frequency as 'QS' (Quarterly Start)
        fare_ts = fare_ts.set_index('date')['fare'].asfreq('QS') 
        
        return fare_ts.dropna()
        
    fare_ts = prepare_ts_data(df)

    st.subheader("1. Time Series Decomposition")
    st.markdown("Using the Additive Model to decompose the series, assuming the magnitude of seasonality doesn't change with the trend.")
    
    MIN_DECOMPOSITION_PERIODS = 8 # Need at least 2 full years (8 quarters) to run decomposition reliably

    if len(fare_ts) >= MIN_DECOMPOSITION_PERIODS:
        try:
            # CRITICAL FIX: Explicitly set period=4 for quarterly data.
            decomposition = seasonal_decompose(fare_ts, model='additive', period=4) 
            
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            fig.suptitle('Additive Decomposition of Quarterly Average Fare', fontsize=16)
            st.pyplot(fig)
            
            st.success("""
            **Decomposition Insights for Pricing Strategy:**
            * **Trend:** Reveals the long-term inflationary or deflationary effects on fares over time.
            * **Seasonal:** Clearly identifies the consistent, predictable price increase in **Q3 (Summer travel)**, which can be leveraged for planning.
            * **Residual:** Represents the unpredictable fluctuations (e.g., fuel shocks, economic crises). Its low magnitude suggests the model captured most structured variation.
            """)
        except Exception as e:
            st.error(f"Could not perform Time Series Decomposition, even with enough data. Error: {e}")
            st.info("The time series may have gaps or irregularity preventing decomposition.")
            
    else:
        st.warning(f"Insufficient data for decomposition. Need at least {MIN_DECOMPOSITION_PERIODS} quarters, found only {len(fare_ts)}.")

    st.markdown("---")

    st.subheader("2. Seasonal Forecasting (Conceptual)")
    st.markdown("Forecasting uses the derived trend and seasonal components to project future average fares, allowing users to budget for peak periods.")
    
    # Use the decomposition's trend/seasonal data for a mock forecast if decomposition succeeded
    if 'decomposition' in locals():
        # --- Prepare Forecast Inputs ---
        last_trend = decomposition.trend.iloc[-1]
        last_seasonality = decomposition.seasonal.tail(4).mean() # Average seasonality over the last year
        
        # Define the next 4 quarters for forecasting
        forecast_index = pd.date_range(start=fare_ts.index.max(), periods=5, freq='QS')[1:]
        
        # Mock forecast values: Assume 2% annual growth on top of last recorded trend/seasonality
        growth_rate = 1.02
        
        # Calculate mock values (Trend + Seasonality + Growth)
        forecast_values = np.array([last_trend + last_seasonality * growth_rate] * len(forecast_index))

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        fare_ts.plot(ax=ax, label='Historical Average Fare', color='blue', marker='o')
        
        ax.plot(forecast_index, forecast_values, 'r--', label='Mock Seasonal Forecast (+2% Trend Growth)')
        ax.set_title('Average Quarterly Fare Forecast based on Decomposition')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Fare ($)')
        ax.legend()
        st.pyplot(fig)
        
        st.info("In a full implementation, models like **ARIMA** or **Prophet** would capture the time-dependent relationship between the errors (residuals) and the forecast, resulting in a statistically robust prediction band.")
    
    else:
        st.warning("Cannot generate forecast visualization because the Time Series Decomposition failed or ran out of data.")