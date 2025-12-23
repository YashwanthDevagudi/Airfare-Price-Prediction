import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


st.title("🧬 Phase 2: Feature Engineering and Factor Analysis")
st.markdown("Creating the temporal and route-specific features necessary for the predictive model, and assessing their relationship with the target variable (Fare).")

if 'df' not in st.session_state:
    st.warning("Please navigate to 'Dataset & EDA' first to load and clean the data.")
else:
    df = st.session_state['df'].copy()

    # --- Feature Engineering (Temporal/Seasonal) ---
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['quarter'].astype(str).apply(lambda x: str(int(x)*3)), format='%Y-%m')
    df['Month'] = df['date'].dt.month
    
    st.subheader("1. Temporal and Seasonal Features")
    st.markdown("We focus on annual and quarterly trends, crucial for identifying seasonal pricing patterns.")

    # --- REWRITTEN SECTION: Seasonal Fare Trend (Quarterly) ---
    st.markdown("**Seasonal Fare Trend (Quarterly)**")
    
    # Calculate the median fare for each quarter across all years
    quarterly_fare = df.groupby('quarter')['fare'].median().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a Line Plot (or Point Plot) to emphasize the sequential trend
    sns.pointplot(
        x='quarter', 
        y='fare', 
        data=df, # Use the full dataframe for bootstrapped error bars (95% CI)
        estimator=np.median, # Use median to be less sensitive to outliers
        capsize=0.1,         # Add caps to the error bars
        color='darkblue', 
        linestyles='--',
        ax=ax
    )
    
    # Adding a simple line for visual flow (optional, but cleaner)
    sns.lineplot(
        x='quarter',
        y='fare',
        data=quarterly_fare,
        ax=ax,
        color='darkblue',
        linewidth=2
    )

    ax.set_title('Seasonal Median Fare Trend (Quarterly)')
    ax.set_xlabel('Quarter (1=Jan-Mar, 4=Oct-Dec)')
    ax.set_ylabel('Median Fare ($)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    st.info("The **Line Plot** clearly shows a pronounced upward trend leading into Q3 (Summer), followed by a sharp drop in Q4. This confirms predictable seasonal high and low points.")
    
    # --- Annual Fare Trend (remains the same) ---
    st.markdown("**Annual Fare Trend**")
    avg_fare_year = df.groupby('Year')['fare'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='fare', data=avg_fare_year, marker='o', ax=ax)
    ax.set_title('Average Fare by Year (1993 - 2024)')
    st.pyplot(fig)
    
    st.markdown("---")

    # Average Fare Over Time (Trend)
    st.markdown("**Annual Fare Trend**")
    avg_fare_year = df.groupby('Year')['fare'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='Year', y='fare', data=avg_fare_year, marker='o', ax=ax)
    ax.set_title('Average Fare by Year (1993 - 2024)')
    st.pyplot(fig)
    
    st.markdown("---")

    st.subheader("2. Market Share and Competition Factors")
    st.markdown("Market share and low-fare carrier presence are key indicators of competition.")
    
    # Large Carrier vs Low-Fare Carrier Fare Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='fare_lg', y='fare_low', data=df, ax=ax, alpha=0.5)
    ax.plot([df['fare'].min(), df['fare'].max()], [df['fare'].min(), df['fare'].max()], 'r--', alpha=0.7, label='Equal Fare Line')
    ax.set_title('Comparison: Large Carrier Fare vs. Low-Fare Carrier Fare')
    ax.legend()
    st.pyplot(fig)
    st.success("The majority of points lie below the red line, indicating the low-fare carrier typically offers a cheaper alternative for the same route.")

    st.markdown("---")

    st.subheader("3. Feature Importance (Correlation-Based)")
    st.markdown("Initial assessment of how key numerical features correlate with the **Fare**.")

    numerical_features = ['fare', 'nsmiles', 'passengers', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low', 'Year']
    corr_matrix = df[numerical_features].corr()
    fare_corr = corr_matrix['fare'].sort_values(ascending=False).drop('fare')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=fare_corr.values, y=fare_corr.index, ax=ax, palette='coolwarm')
    ax.set_title('Correlation with Fare')
    ax.set_xlabel('Pearson Correlation Coefficient')
    st.pyplot(fig)
    
    st.warning(f"**Highest Correlation:** The strongest predictors are, unsurprisingly, related to **competitor fares (fare_lg, fare_low)** and **Distance (nsmiles)**.")
    
    # Store the latest df with new temporal features
    st.session_state['df_features'] = df