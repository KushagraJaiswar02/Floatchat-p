import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta
from timedepthplot import show_time_depth_plot
from map_page import show_map
from chatbot_ui import show_chatbot_ui


st.set_page_config(
    page_title="ARGO-FloatChat",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;   
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-attachment: fixed;
        min-height: 100vh;
        color: #ffffff;
        background-image: url("https://images.unsplash.com/photo-1604599340287-2042e85a3802?q=80&w=774&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;

       
    }

    

    /* Transparent navbar buttons */
div.stButton > button:first-child {
    background: transparent;
    color: #ffffff;            /* white text */
    font-weight: 600;
    font-size: 18px;
    padding: 8px 20px;
    border: none;
    border-bottom: 2px solid transparent;  /* invisible by default */
    cursor: pointer;
    transition: all 0.3s ease;
}

/* Hover effect - white underline */
div.stButton > button:first-child:hover {
    border: 2px solid #ffffff; 
    color: #ffffff;                   /* keep text white */
}

/* Active/selected effect */
div.stButton > button:first-child:active {
    border-bottom: 2px solid #ffffff;  /* solid white underline */
    color: #ffffff;
}

    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 4rem 0;
        margin-bottom: 3rem;
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff, #a8c8ec);
        -webkit-background-clip: text;
        background-clip: text;
        margin-bottom: 1rem;
        
    }

    .hero-subtitle {
        font-size: 1.5rem;
        font-weight: 400;
        color: #ffff;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    .hero-description {
        font-size: 1.1rem;
        color: #ffff;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Feature Cards */
    .feature-card {
        background: rgba(0, 35, 102,0.4);
        backdrop-filter: blur(200px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }

    .feature-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .feature-description {
        color: #d1e3f5;
        line-height: 1.6;
        font-size: 1rem;
    }

    .feature-icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #4fc3f7, #29b6f6);
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
    }

    /* Page Title Styling */
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    /* Content Cards */
    .content-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    /* Statistics Cards */
    .stat-card {
        background: rgba(0, 35, 102,0.9);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4fc3f7;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #b8d4f0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar Styling */
    .sidebar-content {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }

    /* Custom Selectbox and Input Styling */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
    }

    .stSelectbox > div > div > div > div {
        color: white;
    }

    /* Plot Container */
    .plot-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.2rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
    }
    
    /* Text Colors */
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

@st.cache_resource
def create_dummy_data():
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
    
    cursor.execute("SELECT COUNT(*) FROM profiles")
    if cursor.fetchone()[0] == 0:
        np.random.seed(42)
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        lat, lon = 12.9716, 77.5946
        rows = []
        for i in range(100):
            t = base_time + timedelta(days=i*3)
            depth = np.random.choice([0, 10, 20, 50, 100, 200])
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34 + np.random.rand() * 2
            temperature = 15 + np.random.rand() * 10
            air_temp = 20 + np.random.rand() * 5
            oxygen = 200 + np.random.rand() * 50
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))

        base_time_2025 = datetime(2025, 1, 1, 0, 0, 0)
        for i in range(100):
            t = base_time_2025 + timedelta(days=i*3)
            depth = np.random.choice([0, 10, 20, 50, 100, 200])
            latitude = lat + np.random.randn() * 0.1
            longitude = lon + np.random.randn() * 0.1
            salinity = 34.5 + np.random.rand() * 2
            temperature = 16 + np.random.rand() * 10
            air_temp = 21 + np.random.rand() * 5
            oxygen = 210 + np.random.rand() * 50
            rows.append((t, depth, latitude, longitude, salinity, temperature, air_temp, oxygen))

        cursor.executemany("""
        INSERT INTO profiles (time, depth, latitude, longitude, salinity, temperature, air_temp, oxygen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        
        conn.commit()
    return conn

@st.cache_data
def load_data():
    conn = create_dummy_data()
    df = pd.read_sql_query("SELECT * FROM profiles", conn)
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    return df

def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()


nav_cols = st.columns([1, 1, 1, 1, 1, 1])
nav_items = [
    ("", "Home", 'home'),
    ("", "FloatChat", 'chatbot'),
    ("", "Map", 'map'),
    ("", " Profile Comparision", 'comparison'),
    ("", "Depth-Time Plots", 'time_depth')
]

for i, (icon, label, page) in enumerate(nav_items):
    with nav_cols[i+1]:  # Skip first column for spacing
        if st.button(f"{icon} {label}", use_container_width=True, key=f"nav_{page}"):
            navigate_to(page)
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.current_page == "home":
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">ARGO FloatChat</div>
        <div class="hero-subtitle">Analyse Ocean data with ease</div>
        <div class="hero-description">
            Analyse ARGO Float data with simple commands, talk to our chatbot and get real time updates along with visulizations of the required profiles, everything at one place.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                FloatChat AI
            </div>
            <div class="feature-description">
                Intelligent chatbot that makes ocean data accessible to everyone. Query float locations, 
                temperature trends, and salinity variations using natural language. Perfect for researchers 
                and educators alike.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                Smart Analytics
            </div>
            <div class="feature-description">
                Advanced statistical analysis and anomaly detection. Compare multiple datasets, 
                identify trends, and generate comprehensive reports with automated insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                Interactive Mapping
            </div>
            <div class="feature-description">
                Real-time visualization of ARGO float trajectories across global oceans. 
                Interactive maps with zoom controls, temporal filters, and detailed float information 
                for comprehensive monitoring.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                Real-time Data
            </div>
            <div class="feature-description">
                Access to live ARGO float data streams with automatic updates. 
                Monitor ocean conditions as they happen with minimal latency.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                Advanced Visualization
            </div>
            <div class="feature-description">
                Depth-time heatmaps, profile comparisons, and multi-dimensional analysis tools. 
                Explore temperature, salinity, and oxygen variations across different depths 
                and time periods with interactive charts.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">
                <div class="feature-icon"></div>
                Research Tools
            </div>
            <div class="feature-description">
                Comprehensive suite of tools for oceanographic research including 
                profile comparison, statistical analysis, and export capabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #ffffff; margin-bottom: 2rem;'>Platform Statistics</h2>", unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">200+</div>
            <div class="stat-label">Active Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">1M+</div>
            <div class="stat-label">Data Points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">24/7</div>
            <div class="stat-label">Monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">Global</div>
            <div class="stat-label">Coverage</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "chatbot":
    st.markdown('<div class="</div>', unsafe_allow_html=True)
    show_chatbot_ui()

elif st.session_state.current_page == "map":
    st.markdown('<div class="page-title">ARGO float location</div>', unsafe_allow_html=True)
    show_map()

elif st.session_state.current_page == "comparison":
    st.markdown('<div class="page-title">Profile Comparison Analysis</div>', unsafe_allow_html=True)
    df = load_data()
    main_col, sidebar_col = st.columns([0.7, 0.3])

    with sidebar_col:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### Analysis Controls")
        available_years = sorted(df['year'].unique())
        properties = ['salinity', 'temperature', 'air_temp', 'oxygen']
        property_labels = {
            'salinity': ' Salinity',
            'temperature': ' Temperature', 
            'air_temp': ' Air Temperature',
            'oxygen': 'Oxygen'
        }
        
        selected_property = st.selectbox("Choose Profile Type", properties, 
                                       format_func=lambda x: property_labels[x])
        
        st.markdown("**Select Years to Compare**")
        col_from, col_to = st.columns(2)
        with col_from:
            year1 = st.selectbox("From Year", available_years, index=0)
        with col_to:
            year2 = st.selectbox("To Year", available_years, index=1 if len(available_years) > 1 else 0)
        st.markdown('</div>', unsafe_allow_html=True)

    with main_col:
        df_year1 = df[df['year'] == year1]
        df_year2 = df[df['year'] == year2]

        
        st.subheader(f" {selected_property.title()} Comparison: {year1} vs {year2}")
        combined_df = pd.concat([df_year1.assign(year_label=f'{year1}'), df_year2.assign(year_label=f'{year2}')])
        fig_combined = px.line(combined_df, x='time', y=selected_property, color='year_label',
                               title=f"{selected_property.title()} Time Series Comparison",
                               color_discrete_map={f'{year1}':'#4fc3f7', f'{year2}':'#f06292'})
        
        st.plotly_chart(fig_combined, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        
        st.subheader(" Distribution Analysis")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df_year1[selected_property], name=f'{year1}', 
                                boxpoints='outliers', marker_color='#4fc3f7'))
        fig_box.add_trace(go.Box(y=df_year2[selected_property], name=f'{year2}', 
                                boxpoints='outliers', marker_color='#f06292'))
        fig_box.update_layout(
            title=f"{selected_property.title()} Distribution Comparison", 
            yaxis_title=selected_property.title(),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Individual Year Analysis")
        year_col1, year_col2 = st.columns(2)
        with year_col1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig1 = px.line(df_year1, x='time', y=selected_property, 
                          title=f"{selected_property.title()} - {year1}", 
                          color_discrete_sequence=['#4fc3f7'])
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with year_col2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig2 = px.line(df_year2, x='time', y=selected_property, 
                          title=f"{selected_property.title()} - {year2}", 
                          color_discrete_sequence=['#f06292'])
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Statistical Summary")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown(f"** {year1} Statistics**")
            st.write(f"**Mean:** {df_year1[selected_property].mean():.2f}")
            st.write(f"**Std Dev:** {df_year1[selected_property].std():.2f}")
            st.write(f"**Min:** {df_year1[selected_property].min():.2f}")
            st.write(f"**Max:** {df_year1[selected_property].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with stats_col2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown(f"** {year2} Statistics**")
            st.write(f"**Mean:** {df_year2[selected_property].mean():.2f}")
            st.write(f"**Std Dev:** {df_year2[selected_property].std():.2f}")
            st.write(f"**Min:** {df_year2[selected_property].min():.2f}")
            st.write(f"**Max:** {df_year2[selected_property].max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "time_depth":
    st.markdown('<div class="page-title">Depth-Time Analysis</div>', unsafe_allow_html=True)
    show_time_depth_plot()

st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
