# --- app.py (Rewritten for Advanced Flight Price Prediction Project) ---
import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
import os

# Set broad page configuration for better visualization of plots/maps
st.set_page_config(
    page_title="Flight Price Prediction & Route Ranking", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar navigation ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1579737197170-c7526939b434", width=250, caption="Flight Price Analytics")
    
    selected = option_menu(
        menu_title="Project Phases",
        options=[
            "1. Data Acquisition & EDA",
            "2. Feature Engineering",
            "3. Model Training",
            "4. Time Series Analysis",
            "5. Prediction & Route Ranking",
            "6. Route Network Visualization"
        ],
        icons=["database", "cpu", "robot", "graph-up", "trophy", "globe"], # Updated icons
        menu_icon="plane-fill",
        default_index=0
    )

# --- Function to load a page dynamically ---
def load_page(page_filename):
    """Dynamically loads and executes the content of the selected Streamlit page file."""
    # Construct the path to the page file
    page_path = os.path.join("pages", page_filename)
    
    # Check if the file exists before attempting to load
    if not os.path.exists(page_path):
        st.error(f"Error: Page file not found at {page_path}. Please ensure your file structure is correct.")
        return

    try:
        # Use importlib to load the module (page file)
        spec = importlib.util.spec_from_file_location("page", page_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        st.error(f"An error occurred while loading the page: {e}")

# --- Map selected menu to page files ---
# Ensure these filenames match your actual structure precisely
page_files = {
    "1. Data Acquisition & EDA": "1_dataset_eda.py",
    "2. Feature Engineering": "2_feature_engineering.py",
    "3. Model Training": "3_model_training.py",
    "4. Time Series Analysis": "4_time_series_analysis.py",
    "5. Prediction & Route Ranking": "5_prediction_ranking.py",
    "6. Route Network Visualization": "6_route_visualization.py"
}

# --- Load the selected page ---
if selected in page_files:
    load_page(page_files[selected])
else:
    st.error("Invalid page selection.")