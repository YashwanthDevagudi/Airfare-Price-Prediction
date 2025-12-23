import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import os

# --- Metrics Function ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate the Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero, though fares should always be > 0
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Training and Evaluation ---

def train_regression_model(X, y, test_size=0.3, random_state=42):
    """
    Splits data, trains the Random Forest model, and evaluates performance.
    
    Returns: model, metrics_dict
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # --- Model Definition ---
    # Using small parameters for faster execution in a Streamlit dashboard environment
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5,
        random_state=random_state, 
        n_jobs=-1
    )
    
    # --- Training ---
    model.fit(X_train, y_train)
    
    # --- Evaluation ---
    y_pred = model.predict(X_test)
    
    metrics = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'Feature_Importances': pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    }
    
    return model, metrics

# --- Model Persistence (for reuse across sessions) ---

MODEL_FILENAME = 'random_forest_fare_model.joblib'

def save_model(model, path=MODEL_FILENAME):
    """Saves the trained model to a file."""
    try:
        dump(model, path)
        print(f"Model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(path=MODEL_FILENAME):
    """Loads the trained model from a file."""
    if os.path.exists(path):
        return load(path)
    return None