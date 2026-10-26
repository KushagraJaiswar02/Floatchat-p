"""
tests/test_map_figure.py — Unit test to verify Plotly map figure construction with MapLibre API.
"""

import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class TestMapFigureConstruction(unittest.TestCase):
    """Verify MapLibre API (Scattermap) and fallback behavior."""

    def setUp(self):
        self.df = pd.DataFrame({
            "float_id": [1001, 1001, 1002],
            "latitude": [12.5, 12.8, -5.2],
            "longitude": [65.4, 65.9, 80.1],
            "time": ["2023-01-01", "2023-01-02", "2023-01-01"],
            "cumulative_distance_km": [0.0, 45.2, 0.0],
            "temperature": [28.5, 28.2, 29.1],
            "salinity": [35.2, 35.3, 34.9],
            "depth": [10.0, 15.0, 5.0],
            "oxygen": [200.0, 198.0, 205.0],
        })

    def test_map_page_figure_constructs_successfully(self):
        """Construct the trajectory figure using the exact logic from frontend/map_page.py."""
        filtered_df = self.df
        fig_map = go.Figure()
        color_list = px.colors.qualitative.Set1

        scatter_map_fn = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))
        self.assertIsNotNone(scatter_map_fn, "Neither Scattermap nor Scattermapbox is available")

        for i, float_id in enumerate(filtered_df["float_id"].unique()):
            float_df = filtered_df[filtered_df["float_id"] == float_id].sort_values("time")
            customdata = np.array(float_df[["cumulative_distance_km"]].round(2))

            fig_map.add_trace(scatter_map_fn(
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
                text=float_df["time"].astype(str),
            ))

        map_config = dict(
            style="open-street-map",
            zoom=3,
            center={"lat": float(filtered_df["latitude"].mean()), "lon": float(filtered_df["longitude"].mean())},
        )
        layout_kwargs = {
            "margin": {"r": 0, "t": 30, "l": 0, "b": 0},
            "title": "ARGO Float Trajectories with Distance Traveled",
        }
        if hasattr(go, "Scattermap"):
            layout_kwargs["map"] = map_config
        else:
            layout_kwargs["mapbox"] = map_config

        fig_map.update_layout(**layout_kwargs)

        # Assert traces were added and layout matches
        self.assertEqual(len(fig_map.data), 2)
        if hasattr(go, "Scattermap"):
            self.assertEqual(fig_map.layout.map.style, "open-street-map")
        else:
            self.assertEqual(fig_map.layout.mapbox.style, "open-street-map")

    def test_dummy_scatter_map_constructs_successfully(self):
        """Construct the scatter map using the logic in frontend/dummy.py."""
        scatter_map_func = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
        self.assertIsNotNone(scatter_map_func)

        map_style_kw = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"
        map_kwargs = {
            "lat": "latitude",
            "lon": "longitude",
            "color": "temperature",
            "size": "depth",
            "hover_data": ["salinity", "oxygen"],
            map_style_kw: "open-street-map",
            "zoom": 8,
            "height": 600,
            "title": "ARGO Float Locations",
        }
        fig = scatter_map_func(self.df, **map_kwargs)
        self.assertIsNotNone(fig)
        self.assertTrue(len(fig.data) > 0)


if __name__ == "__main__":
    unittest.main()
