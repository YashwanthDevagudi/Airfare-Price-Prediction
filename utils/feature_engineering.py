import pandas as pd
import numpy as np

def create_temporal_features(df):
    """
    Creates temporal features based on Year and Quarter.
    (Simulates Days Until Departure / Booking Window for a richer model,
     as actual booking date is missing from this specific dataset.)
    """
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['quarter'].astype(str).apply(lambda x: str(int(x)*3)), format='%Y-%m')
    df['Month'] = df['date'].dt.month
    
    # Simulate Days Until Departure (Booking Window) for the model
    # Assumption: Longest routes are booked further out, and there's inherent randomness.
    df['days_until_departure'] = np.random.randint(low=20, high=180, size=len(df))
    
    # Assume Q3 (Summer) is booked earliest, Q4 (Fall/Winter) is booked later on average
    df.loc[df['quarter'] == 3, 'days_until_departure'] = df.loc[df['quarter'] == 3, 'days_until_departure'].apply(lambda x: x + np.random.randint(30, 60))
    
    # Use log transform for this simulated feature to normalize its distribution
    df['log_days_until_departure'] = np.log1p(df['days_until_departure'])

    return df

def create_route_characteristics(df):
    """
    Aggregates data to create route-level features:
    - Average passenger count (Route Popularity)
    - Average distance (nsmiles)
    """
    
    # Create unique route identifier, ensuring A-B is the same as B-A for analysis
    def get_standard_route(row):
        cities = sorted([row['city1'], row['city2']])
        return f"{cities[0]}-{cities[1]}"
        
    df['Route_ID'] = df.apply(get_standard_route, axis=1)

    # Calculate route-level stats
    route_stats = df.groupby('Route_ID').agg(
        avg_route_passengers=('passengers', 'mean'),
        total_route_passengers=('passengers', 'sum'),
        avg_nsmiles=('nsmiles', 'mean'),
    ).reset_index()

    # Merge stats back to the original DataFrame
    df = df.merge(route_stats, on='Route_ID', how='left')
    
    # Feature scaling for route popularity
    df['log_avg_passengers'] = np.log1p(df['avg_route_passengers'])

    return df

def run_feature_engineering_pipeline(df):
    """Executes all feature engineering steps."""
    df = create_temporal_features(df)
    df = create_route_characteristics(df)
    
    # Prepare data for modeling by selecting and encoding final features
    
    # Select features based on strong correlation and project requirements
    MODEL_FEATURES = [
        'nsmiles', 'log_avg_passengers', 'fare_lg', 'fare_low', 'large_ms', 'lf_ms', 
        'log_days_until_departure'
    ]
    CATEGORICAL_FEATURES = ['quarter', 'Year']
    
    # Select a smaller subset for modeling speed in the dashboard
    df_model = df[MODEL_FEATURES + CATEGORICAL_FEATURES + ['fare']].copy()

    # One-Hot Encode Categorical Features
    df_model = pd.get_dummies(df_model, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    df_model.dropna(inplace=True) # Final check for any NaNs introduced by merge/encoding

    X = df_model.drop('fare', axis=1)
    y = df_model['fare']
    
    return X, y, df_model