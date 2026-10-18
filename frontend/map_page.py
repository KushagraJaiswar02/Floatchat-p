import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go

def show_map():
  
    np.random.seed(42)
    n_floats = 5
    n_points = 50
    base_coords = [(12.0, 80.0), (15.0, 82.0), (10.0, 78.0), (18.0, 85.0), (20.0, 88.0)]

    start_time = datetime(2020, 1, 1)
    end_time = datetime(2025, 12, 31)

    rows = []
    for float_id in range(n_floats):
        lat0, lon0 = base_coords[float_id]
        for i in range(n_points):
            t = start_time + (end_time - start_time) * np.random.rand()
            latitude = lat0 + np.random.randn() * 0.1
            longitude = lon0 + np.random.randn() * 0.1
            depth = np.random.choice([0, 10, 20, 50, 100, 200])
            temperature = 15 + np.random.rand() * 10
            salinity = 34 + np.random.rand() * 2
            oxygen = 200 + np.random.rand() * 50
            rows.append([float_id, t, latitude, longitude, depth, temperature, salinity, oxygen])

    df = pd.DataFrame(rows, columns=["float_id", "time", "latitude", "longitude", "depth", "temperature", "salinity", "oxygen"])
    df.sort_values(by=["float_id", "time"], inplace=True)

   
    def compute_cumulative_distance(df):
        df = df.copy()
        df["cumulative_distance_km"] = 0.0
        for float_id in df["float_id"].unique():
            float_df = df[df["float_id"] == float_id].sort_values("time")
            distances = [0]
            for i in range(1, len(float_df)):
                prev = (float_df.iloc[i - 1]["latitude"], float_df.iloc[i - 1]["longitude"])
                curr = (float_df.iloc[i]["latitude"], float_df.iloc[i]["longitude"])
                distances.append(geodesic(prev, curr).km)
            df.loc[df["float_id"] == float_id, "cumulative_distance_km"] = np.cumsum(distances)
        return df

    df = compute_cumulative_distance(df)

   
    left, right = st.columns([4, 2])

    with right:
        st.header("Filters")

        float_options = df["float_id"].unique()
        selected_float = st.multiselect("Select Float(s):", float_options, default=float_options)
        filtered_df = df[df["float_id"].isin(selected_float)]

        st.subheader("Date & Time Range")
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From Date", filtered_df["time"].min().date())
            time_from = st.time_input("From Time", filtered_df["time"].min().time())
        with col2:
            date_to = st.date_input("To Date", filtered_df["time"].max().date())
            time_to = st.time_input("To Time", filtered_df["time"].max().time())
        datetime_from = datetime.combine(date_from, time_from)
        datetime_to = datetime.combine(date_to, time_to)
        filtered_df = filtered_df[(filtered_df["time"] >= datetime_from) & (filtered_df["time"] <= datetime_to)]

        st.subheader("Depth Range (m)")
        if not filtered_df.empty:
            depth_min, depth_max = float(filtered_df["depth"].min()), float(filtered_df["depth"].max())
            depth_from, depth_to = st.slider(
                "Select Depth Range:", min_value=depth_min, max_value=depth_max, value=(depth_min, depth_max)
            )
            filtered_df = filtered_df[(filtered_df["depth"] >= depth_from) & (filtered_df["depth"] <= depth_to)]
        else:
            st.warning("No data available for selected floats and date range.")

        st.subheader("Choose Graph Axes")
        col3, col4 = st.columns(2)
        axis_options = ["temperature", "salinity", "oxygen", "depth", "latitude", "longitude", "time", "cumulative_distance_km"]
        with col3:
            x_axis = st.selectbox("X-axis", axis_options, index=0)
        with col4:
            y_axis = st.selectbox("Y-axis", axis_options, index=3)

        st.subheader("Nearest Float Lookup")
        col5, col6 = st.columns(2)
        with col5:
            lat_input = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0)
        with col6:
            lon_input = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=82.0)

        def find_nearest(df, lat, lon):
            df = df.copy()
            df["distance_km"] = df.apply(lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km, axis=1)
            return df.nsmallest(3, "distance_km")

        if not filtered_df.empty:
            nearest_floats = find_nearest(filtered_df, lat_input, lon_input)
            st.write("Nearest Floats:")
            st.dataframe(nearest_floats[["float_id", "latitude", "longitude", "distance_km"]])
        else:
            st.info("No nearest floats to show.")

    with left:
        if not filtered_df.empty:
            st.subheader("ARGO Float Trajectories on Ocean Map")
            fig_map = go.Figure()
            color_list = px.colors.qualitative.Set1  # Distinct colors per float

            for i, float_id in enumerate(filtered_df["float_id"].unique()):
                float_df = filtered_df[filtered_df["float_id"] == float_id].sort_values("time")
                customdata = np.array(float_df[["cumulative_distance_km"]].round(2))

                fig_map.add_trace(go.Scattermapbox(
                    lon=float_df["longitude"],
                    lat=float_df["latitude"],
                    mode="lines+markers",
                    marker=dict(size=8, color=color_list[i % len(color_list)]),
                    line=dict(width=2, color=color_list[i % len(color_list)]),
                    name=f"Float {float_id}",
                    customdata=customdata,
                    hovertemplate=(
                        "Time: %{text}<br>"
                        "Lat: %{lat}<br>"
                        "Lon: %{lon}<br>"
                        "Distance Traveled (km): %{customdata[0]}<extra></extra>"
                    ),
                    text=float_df["time"].astype(str)
                ))

            fig_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    zoom=3,
                    center={"lat": filtered_df["latitude"].mean(), "lon": filtered_df["longitude"].mean()},
                ),
                margin={"r":0,"t":30,"l":0,"b":0},
                title="ARGO Float Trajectories with Distance Traveled"
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("Custom Scatter Plot")
            fig_scatter = px.scatter(
                filtered_df,
                x=x_axis,
                y=y_axis,
                color="float_id",
                hover_data=["time", "temperature", "salinity", "oxygen", "depth", "cumulative_distance_km"],
                title=f"{y_axis.capitalize()} vs {x_axis.capitalize()}"
            )
            if y_axis == "depth":
                fig_scatter.update_yaxes(autorange="reversed")
            if x_axis == "depth":
                fig_scatter.update_xaxes(autorange="reversed")
            st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander(" Show Raw Data"):
                st.dataframe(filtered_df)
        else:
            st.warning("No data available for the selected filters.")