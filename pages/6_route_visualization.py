# --- pages/6_route_visualization.py ---
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

st.set_page_config(page_title="Flight Route Visualization", layout="wide")
st.title("✈️ Phase 6: Flight Route Visualization and Analytics")
st.markdown(
    "Visualizing US flight routes interactively. Route thickness = passenger volume, route color = average fare, popups show city names."
)
st.markdown("---")

# -------------------------------
# Load local CSV
# -------------------------------
DATA_PATH = "data/flight_data.csv"  # Adjust path if needed

@st.cache_data
def load_data(file_path):
    """Load CSV file"""
    df = pd.read_csv(file_path, low_memory=False)
    return df

df_raw = load_data(DATA_PATH)

# -------------------------------
# Helper: parse coordinates
# -------------------------------
def parse_coordinates(coord):
    """Convert '(lat, lon)' string to floats"""
    try:
        coord = coord.strip("()").strip()
        lat, lon = coord.split(",")
        return float(lat), float(lon)
    except:
        return None, None

# -------------------------------
# Preprocess routes
# -------------------------------
@st.cache_data
def preprocess_routes(df):
    # Keep only rows with valid geocoded coordinates
    df = df[df['Geocoded_City1'].str.contains(r'\(.*\)', na=False)]
    df = df[df['Geocoded_City2'].str.contains(r'\(.*\)', na=False)]

    # Parse coordinates
    df[['lat1', 'lon1']] = df['Geocoded_City1'].apply(lambda x: pd.Series(parse_coordinates(x)))
    df[['lat2', 'lon2']] = df['Geocoded_City2'].apply(lambda x: pd.Series(parse_coordinates(x)))

    # Extract city names
    df['City1'] = df['Geocoded_City1'].apply(lambda x: x.split(",")[0].strip())
    df['City2'] = df['Geocoded_City2'].apply(lambda x: x.split(",")[0].strip())

    # Keep relevant columns
    df = df[['lat1','lon1','lat2','lon2','passengers','fare','City1','City2']].dropna()
    return df

df_routes = preprocess_routes(df_raw)

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Map Controls")

fare_range = st.sidebar.slider(
    "Fare Range ($)",
    float(df_routes['fare'].min()),
    float(df_routes['fare'].max()),
    (float(df_routes['fare'].min()), float(df_routes['fare'].max()))
)

passenger_range = st.sidebar.slider(
    "Passenger Count",
    int(df_routes['passengers'].min()),
    int(df_routes['passengers'].max()),
    (int(df_routes['passengers'].min()), int(df_routes['passengers'].max()))
)

max_routes = min(len(df_routes), 5000)
top_n = st.sidebar.slider(
    "Number of Routes to Display",
    100, max_routes, min(1000, max_routes), step=100
)

opacity = st.sidebar.slider("Route Opacity", 0.1, 1.0, 0.5, step=0.1)
color_scheme = st.sidebar.selectbox(
    "Color Scheme",
    ["Reds","Blues","Greens","Purples","Viridis","Plasma"]
)

# -------------------------------
# Filter & sample data
# -------------------------------
df_filtered = df_routes[
    (df_routes['fare'] >= fare_range[0]) & (df_routes['fare'] <= fare_range[1]) &
    (df_routes['passengers'] >= passenger_range[0]) & (df_routes['passengers'] <= passenger_range[1])
]

if len(df_filtered) > top_n:
    df_display = df_filtered.sample(n=top_n, random_state=42)
else:
    df_display = df_filtered

# -------------------------------
# Dataset overview
# -------------------------------
with st.expander("📊 Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df_raw):,}")
    col2.metric("Columns", len(df_raw.columns))
    col3.metric("Memory Usage", f"{df_raw.memory_usage(deep=True).sum()/1024**2:.2f} MB")
    st.dataframe(df_raw.head(10), use_container_width=True)

# -------------------------------
# Generate interactive map
# -------------------------------
st.markdown("### 🗺️ Interactive Flight Routes Map")
st.markdown("Route thickness = passenger volume | Route color = average fare")

with st.spinner("Generating map..."):
    m = folium.Map(location=[39.8283,-98.5795], zoom_start=4, tiles='CartoDB positron')

    norm = matplotlib.colors.Normalize(vmin=df_display['fare'].min(), vmax=df_display['fare'].max())
    cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=color_scheme)

    for _, row in df_display.iterrows():
        color = matplotlib.colors.rgb2hex(cmap.to_rgba(row['fare']))
        weight = 1 + (row['passengers'] / df_display['passengers'].max()) * 5
        popup_text = f"""
        <b>Route Details</b><br>
        Passengers: {int(row['passengers']):,}<br>
        Average Fare: ${row['fare']:.2f}<br>
        From: {row['City1']}<br>
        To: {row['City2']}
        """
        folium.PolyLine(
            locations=[(row['lat1'], row['lon1']), (row['lat2'], row['lon2'])],
            weight=weight,
            color=color,
            opacity=opacity,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    folium_static(m, width=1400, height=600)

# -------------------------------
# Analytics tabs
# -------------------------------
st.markdown("---")
st.markdown("### 📊 Route Analytics")
tab1, tab2, tab3 = st.tabs(["Fare Distribution","Passenger Volume","Fare vs Passengers"])

with tab1:
    fig, ax = plt.subplots(figsize=(12,5))
    sns.histplot(df_display['fare'], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Average Fare ($)")
    ax.set_ylabel("Number of Routes")
    ax.set_title("Fare Distribution")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(12,5))
    sns.histplot(df_display['passengers'], bins=50, kde=True, ax=ax, color='steelblue')
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Number of Routes")
    ax.set_title("Passenger Volume Distribution")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(12,6))
    scatter = ax.scatter(df_display['passengers'], df_display['fare'], c=df_display['fare'], cmap=color_scheme, alpha=0.6, s=50)
    ax.set_xlabel("Passengers")
    ax.set_ylabel("Average Fare ($)")
    ax.set_title("Fare vs Passenger Volume")
    plt.colorbar(scatter, ax=ax, label="Fare ($)")
    st.pyplot(fig)

# -------------------------------
# Download filtered data
# -------------------------------
st.markdown("---")
st.markdown("### 💾 Download Filtered Route Data")
csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", data=csv, file_name="filtered_flight_routes.csv", mime="text/csv")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray'>Flight Route Visualization Dashboard | Data-driven insights for aviation analytics</div>",
    unsafe_allow_html=True
)
