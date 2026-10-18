import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="ARGO - Planet Earth",
    
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Custom CSS for styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove padding from main container */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Full height background */
    .stApp {
        
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Navigation bar styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 40px;
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
      
        margin-bottom: 20px;
    }
    
    .logo {
        color: white;
        font-size: 24px;
        font-weight: bold;
        letter-spacing: 2px;
    }
    
    .nav-links {
        display: flex;
        gap: 10px;
        align-items: centre;
    }
    
    /* Main content styling for home page */
    .main-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 70vh;
        color: white;
        padding: 0 20px;
    }
    
    .planet-title {
        font-size: 16px;
        font-weight: 300;
        letter-spacing: 4px;
        margin-bottom: 20px;
        opacity: 0.8;
    }
    
    .earth-title {
        font-size: 80px;
        font-weight: 300;
        letter-spacing: 8px;
        margin-bottom: 40px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .description {
        max-width: 600px;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 40px;
        opacity: 0.9;
    }
    
    /* Page content styling */
    .page-content {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        margin: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling for comparison page */
    .comparison-sidebar {
        background: rgba(0, 0, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
        height: fit-content;
        position: sticky;
        top: 20px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .navbar {
            flex-direction: column;
            gap: 10px;
            padding: 20px;
        }
        
        .earth-title {
            font-size: 50px;
            letter-spacing: 4px;
        }
        
        .description {
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to create dummy database and data
@st.cache_resource
def create_dummy_data():
    """Create dummy database with sample data"""
    conn = sqlite3.connect("dummy.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time TIMESTAMP,
        depth REAL,
        latitude REAL,
        longitude REAL,
        salinity REAL,
        temperature REAL,
        air_temp REAL,
        oxygen REAL
    )
    """)
    
    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM profiles")
    if cursor.fetchone()[0] == 0:
        np.random.seed(42)
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        lat, lon = 12.9716, 77.5946 
        
        rows = []
        
        # Generate data for 2021
        base_time_2021 = datetime(2021, 1, 1, 0, 0, 0)
        for i in range(200):  
            t = base_time_2021 + timedelta(hours=i)
            depth = np.random.choice([0, 10, 20, 50, 100, 200]) 
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2  
            temperature = 15 + np.random.rand() * 10  
            air_temp = 20 + np.random.rand() * 5   
            oxygen = 200 + np.random.rand() * 50  
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))
        
        # Generate data for 2022
        base_time_2022 = datetime(2022, 1, 1, 0, 0, 0)
        for i in range(200):  
            t = base_time_2022 + timedelta(hours=i)
            depth = np.random.choice([0, 10, 20, 50, 100, 200]) 
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2  
            temperature = 15 + np.random.rand() * 10  
            air_temp = 20 + np.random.rand() * 5   
            oxygen = 200 + np.random.rand() * 50  
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))
        
        # Generate data for 2023
        base_time_2023 = datetime(2023, 1, 1, 0, 0, 0)
        for i in range(200):  
            t = base_time_2023 + timedelta(hours=i)
            depth = np.random.choice([0, 10, 20, 50, 100, 200]) 
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2  
            temperature = 15 + np.random.rand() * 10  
            air_temp = 20 + np.random.rand() * 5   
            oxygen = 200 + np.random.rand() * 50  
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))
        
        # Generate data for 2024
        base_time_2024 = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(200):  
            t = base_time_2024 + timedelta(hours=i)
            depth = np.random.choice([0, 10, 20, 50, 100, 200]) 
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2  
            temperature = 15 + np.random.rand() * 10  
            air_temp = 20 + np.random.rand() * 5   
            oxygen = 200 + np.random.rand() * 50  
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))
        
        # Generate data for 2025 (using your exact structure)
        for i in range(200):  
            t = base_time + timedelta(hours=i)
            depth = np.random.choice([0, 10, 20, 50, 100, 200]) 
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2  
            temperature = 15 + np.random.rand() * 10  
            air_temp = 20 + np.random.rand() * 5   
            oxygen = 200 + np.random.rand() * 50  
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))
        
        cursor.executemany("""
        INSERT INTO profiles (time, depth, latitude, longitude, salinity, temperature, air_temp, oxygen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        conn.commit()
    
    return conn

# Load data
@st.cache_data
def load_data():
    """Load data from database"""
    conn = create_dummy_data()
    df = pd.read_sql_query("SELECT * FROM profiles", conn)
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    return df

# Navigation functions
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# Navigation bar with Streamlit buttons
st.markdown('<div class="navbar"><div class="logo">ARGO</div></div>', unsafe_allow_html=True)

# Navigation buttons
col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 1])

with col2:
    if st.button(" Home", use_container_width=True):
        navigate_to('home')

with col3:
    if st.button(" ARGO Float Profile", use_container_width=True):
        navigate_to('profile')

with col4:
    if st.button(" Map Trajectory", use_container_width=True):
        navigate_to('map')

with col5:
    if st.button(" Profile Comparisons", use_container_width=True):
        navigate_to('comparison')

with col6:
    if st.button(" Time Depth Plots", use_container_width=True):
        navigate_to('time_depth')

# Page routing
if st.session_state.current_page == 'home':
    # Home page content
    st.markdown("""
    <div class="main-content">
        <div class="planet-title">PLANET</div>
        <div class="earth-title">EARTH</div>
        <div class="description">
            Learn more about this fascinating miracle that we call our home, Planet Earth. 
            Explore ARGO float data, analyze ocean profiles, and discover the secrets of our oceans.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == 'profile':
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("ARGO Float Profile")
    st.write("Here you can explore individual ARGO float profiles and their measurements.")
    
    # Load data for profile display
    df = load_data()
    
    # Sample profile data
    st.subheader("Sample Profile Data")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Data Summary")
    st.write(df.describe())
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'map':
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title(" Map Trajectory")
    st.write("Visualize the trajectory of ARGO floats across the ocean.")
    
    df = load_data()
    
    # Create map
    fig = px.scatter_mapbox(df, 
                           lat="latitude", 
                           lon="longitude",
                           color="temperature",
                           size="depth",
                           hover_data=["salinity", "oxygen"],
                           mapbox_style="open-street-map",
                           zoom=8,
                           height=600,
                           title="ARGO Float Locations")
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'comparison':
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title(" Profile Comparisons")
    st.write("Compare oceanographic properties between different time periods.")
    
    df = load_data()
    
    # Create two columns: left for graphs (70%), right for controls (30%)
    main_col, sidebar_col = st.columns([0.7, 0.3])
    
    with sidebar_col:
        st.markdown('<div class="comparison-sidebar">', unsafe_allow_html=True)
        st.subheader("Controls")
        
        # Year comparison selector
        available_years = sorted(df['year'].unique())
        
        st.write("**Select Years to Compare:**")
        year1 = st.selectbox("First Year", available_years, index=0, key="year1_select")
        year2 = st.selectbox("Second Year", available_years, index=1 if len(available_years) > 1 else 0, key="year2_select")
        
        # Property selector
        properties = ['salinity', 'temperature', 'air_temp', 'oxygen']
        selected_property = st.selectbox("Choose Parameter", properties, key="property_select")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with main_col:
        # Filter data by years
        df_year1 = df[df['year'] == year1]
        df_year2 = df[df['year'] == year2]
        
        # Combined comparison (main plot)
        st.subheader(f"{selected_property.title()} Comparison: {year1} vs {year2}")
        
        # Prepare data for combined plot
        df_year1_plot = df_year1.copy()
        df_year1_plot['year_label'] = f'{year1}'
        df_year2_plot = df_year2.copy()
        df_year2_plot['year_label'] = f'{year2}'
        
        combined_df = pd.concat([df_year1_plot, df_year2_plot])
        
        fig_combined = px.line(combined_df, 
                              x='time', 
                              y=selected_property,
                              color='year_label',
                              title=f"{selected_property.title()} Time Series Comparison",
                              color_discrete_map={f'{year1}': 'blue', f'{year2}': 'red'})
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Distribution comparison (Box plot)
        st.subheader("Distribution Comparison")
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(y=df_year1[selected_property], 
                                name=f'{year1}', 
                                boxpoints='outliers',
                                marker_color='blue'))
        
        fig_box.add_trace(go.Box(y=df_year2[selected_property], 
                                name=f'{year2}', 
                                boxpoints='outliers',
                                marker_color='red'))
        
        fig_box.update_layout(title=f"{selected_property.title()} Distribution Comparison",
                             yaxis_title=selected_property.title())
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Individual year plots (side by side)
        st.subheader("Individual Year Analysis")
        year_col1, year_col2 = st.columns(2)
        
        with year_col1:
            fig1 = px.line(df_year1, 
                          x='time', 
                          y=selected_property,
                          title=f"{selected_property.title()} - {year1}",
                          color_discrete_sequence=['blue'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with year_col2:
            fig2 = px.line(df_year2, 
                          x='time', 
                          y=selected_property,
                          title=f"{selected_property.title()} - {year2}",
                          color_discrete_sequence=['red'])
            st.plotly_chart(fig2, use_container_width=True)
        
        # Statistics comparison
        st.subheader("Statistical Summary")
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write(f"**{year1} Statistics:**")
            st.write(f"Mean: {df_year1[selected_property].mean():.2f}")
            st.write(f"Std: {df_year1[selected_property].std():.2f}")
            st.write(f"Min: {df_year1[selected_property].min():.2f}")
            st.write(f"Max: {df_year1[selected_property].max():.2f}")
        
        with stats_col2:
            st.write(f"**{year2} Statistics:**")
            st.write(f"Mean: {df_year2[selected_property].mean():.2f}")
            st.write(f"Std: {df_year2[selected_property].std():.2f}")
            st.write(f"Min: {df_year2[selected_property].min():.2f}")
            st.write(f"Max: {df_year2[selected_property].max():.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'time_depth':
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title(" Time Depth Plots")
    st.write("Analyze how properties vary with depth over time.")
    
    df = load_data()
    
    # Property selector
    properties = ['salinity', 'temperature', 'air_temp', 'oxygen']
    selected_property = st.selectbox("Select Property", properties, key="time_depth_property")
    
    # Create time-depth plot
    fig = px.scatter(df, 
                    x='time', 
                    y='depth', 
                    color=selected_property,
                    title=f"{selected_property.title()} vs Time and Depth",
                    color_continuous_scale='viridis')
    
    fig.update_yaxis(autorange='reversed')  # Invert y-axis so depth increases downward
    st.plotly_chart(fig, use_container_width=True)
    
    # Depth profile for selected time range
    st.subheader("Depth Profile Analysis")
    
    # Time range selector
    min_date = df['time'].min().date()
    max_date = df['time'].max().date()
    
    date_range = st.date_input("Select Date Range", 
                              value=(min_date, max_date),
                              min_value=min_date,
                              max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['time'].dt.date >= start_date) & (df['time'].dt.date <= end_date)]
        
        # Average by depth
        depth_avg = filtered_df.groupby('depth')[selected_property].mean().reset_index()
        
        fig_depth = px.line(depth_avg, 
                           x=selected_property, 
                           y='depth',
                           title=f"Average {selected_property.title()} by Depth")
        
        fig_depth.update_yaxis(autorange='reversed')
        st.plotly_chart(fig_depth, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)