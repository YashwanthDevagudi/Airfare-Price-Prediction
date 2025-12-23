import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add utils directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.preprocessing import load_data, clean_data

st.title("🗄️ Phase 1: Data Acquisition and Quality Assessment")
st.markdown("Loading the US Airline Flight Routes and Fares dataset, performing initial cleaning, and assessing data quality.")

# --- Data Loading and Caching ---
@st.cache_data
def get_raw_data():
    return load_data()

@st.cache_data
def get_cleaned_data(df_raw):
    # Pass the raw data to the cleaner to run the full pipeline once
    return clean_data(df_raw)

if 'df_raw' not in st.session_state:
    with st.spinner('Loading raw data...'):
        st.session_state['df_raw'] = get_raw_data()
        
if 'df' not in st.session_state and not st.session_state['df_raw'].empty:
    with st.spinner('Cleaning data (Handling geocodes, NaNs, and outliers)...'):
        st.session_state['df'] = get_cleaned_data(st.session_state['df_raw'])

df_raw = st.session_state.get('df_raw', pd.DataFrame())
df = st.session_state.get('df', pd.DataFrame())

if df_raw.empty:
    st.error("Data could not be loaded. Please check the `DATA_PATH` in `utils/preprocessing.py`.")
else:
    st.subheader("1. Raw Data Overview")
    st.write(f"**Total Records:** {df_raw.shape[0]:,}")
    st.dataframe(df_raw.head())
    st.write("Data Types:")
    st.dataframe(df_raw.dtypes.to_frame(name='Type'))

    st.subheader("2. Missing Value Visualization")
    st.markdown("A heatmap showing null values before cleaning/imputation.")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_raw.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Data Heatmap (Raw Data)')
    st.pyplot(fig)
    
    st.code("Missing values primarily exist in carrier market share and geocoded city columns.")

    st.subheader("3. Cleaning Summary")
    if not df.empty:
        removed_count = df_raw.shape[0] - df.shape[0]
        st.success(f"Cleaning Complete: {df.shape[0]:,} records retained.")
        st.info(f"Total Records Removed (NaNs, Imputation Failures, Outliers): {removed_count:,}")
        st.markdown(f"""
        * **Geocodes:** Imputed using city dictionary where available.
        * **NaNs:** Rows with missing market share/fare data were dropped.
        * **Outliers:** Removed using a Z-score > 3 threshold on numerical columns.
        """)
        st.dataframe(df.describe().T)
    else:
        st.warning("Cleaned data is empty. Check the cleaning function.")