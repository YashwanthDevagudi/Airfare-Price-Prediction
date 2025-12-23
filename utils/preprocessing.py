import pandas as pd
import numpy as np
from scipy import stats
import re

# NOTE: Update this path to where your actual US data is stored.
DATA_PATH = 'data/flight_data.csv   ' 

def remove_endl(s):
    """Helper to clean up geocoded strings."""
    return s.split('\n')[-1] if isinstance(s, str) and s.count('\n') >= 0 else str(s)

def extract_coordinates(geo_str):
    """Parses latitude and longitude from the geocoded string."""
    try:
        # Regex to find two floats separated by comma, handles (lat, lon) format
        coord = re.search(r'(-?\d+\.\d+),\s*(-?\d+\.\d+)', str(geo_str))
        if coord:
            # Returns (lat, lon) as floats
            return float(coord.group(1)), float(coord.group(2)) 
        return None, None
    except:
        return None, None

def load_data(data_path=DATA_PATH):
    """Loads the raw data."""
    try:
        return pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return pd.DataFrame()

def clean_data(df):
    """Performs the full initial cleaning pipeline."""
    df_cleaned = df.copy()
    
    # 1. Geocode Pre-processing
    for col in ['Geocoded_City1', 'Geocoded_City2']:
        df_cleaned[col] = df_cleaned[col].astype(str).apply(remove_endl)

    # 2. Impute missing coordinates (based on city dictionary)
    city_coords = {}
    for city, coord in zip(df_cleaned['city1'], df_cleaned['Geocoded_City1']):
        if coord != 'nan': city_coords[city] = coord
    for city, coord in zip(df_cleaned['city2'], df_cleaned['Geocoded_City2']):
        if coord != 'nan': city_coords[city] = coord

    def fill_coords(row):
        for i in ['1', '2']:
            col = 'Geocoded_City' + i
            if row[col] == 'nan':
                row[col] = city_coords.get(row['city' + i], 'nan')
        return row
        
    df_cleaned = df_cleaned.apply(fill_coords, axis=1)
    
    # 3. Handle remaining NaNs (primarily carrier/market share columns)
    df_cleaned.dropna(inplace=True) 
    
    # 4. Outlier Removal (Z-Score)
    numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    z_scores = np.abs(stats.zscore(df_cleaned[numerical_cols]))
    df_no_outliers = df_cleaned[(z_scores < 3).all(axis=1)]
    
    return df_no_outliers   