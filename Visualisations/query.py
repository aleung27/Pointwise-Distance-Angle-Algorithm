import pandas as pd
from pathlib import Path
import os
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from typing import Tuple
import geoplot
import contextily
import geopy.geocoders.nominatim as nominatim
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.text import Text

DATASET_PATH = os.path.join(Path(__file__).parent.parent, "train.csv")
NEW_YORK_TIF = "new_york.tif"

WEB_MERCATOR = "EPSG:3857"
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude


def get_area() -> Tuple[Point, float]:
    lon_lat = input("Enter a space-separated longitude and latitude: ")
    radius = float(input("Enter a radius (m): "))

    return (Point(float(lon_lat.split()[0]), float(lon_lat.split()[1])), radius)


class NYCMap:
    """
    Controls the underlying map of New York City.
    """

    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        self.fetch_local_tif()
        self.plot(fig=fig, ax=ax)

    def fetch_local_tif(self):
        """
        Fetch the local tif file for the New York map if it doesn't exist yet.
        This downloads a map representation of New York from OSM and saves it locally.
        On further executions, this will not be downloaded again allowing for faster map rendering.
        """
        if not os.path.exists(NEW_YORK_TIF):
            contextily.Place(
                "New York",
                geocoder=nominatim.Nominatim(timeout=100, user_agent="vis-query"),
                zoom=13,
                path=NEW_YORK_TIF,
            )

    def plot(self, fig: plt.Figure, ax: plt.Axes):
        """
        Plot the map of New York onto the provided Axes object.
        """

        # Fetch the New York Boroughs and project to Web Mercator
        boroughs: gpd.GeoDataFrame = gpd.read_file(gpd.datasets.get_path("nybb"))
        boroughs.to_crs(WGS_84, inplace=True)

        # Plot NY and add the OSM base map for the overall display
        boroughs.plot(ax=ax, color="none", linewidth=0)
        contextily.add_basemap(ax=ax, source=NEW_YORK_TIF, crs=WGS_84)


class TaxiQuerier:
    df: pd.DataFrame  # The underlying dataframe for the taxi data
    gdf: gpd.GeoDataFrame  # The transformed df for the taxi data with lat/long geometry column

    gdf_start: gpd.GeoDataFrame  # The gdf for the starting boundary about the point
    gdf_end: gpd.GeoDataFrame  # The gdf for the ending boundary about the point

    def __init__(self):
        self.load_data()
        self.gdf_start = None
        self.gdf_end = None

    def load_data(self):
        self.df = pd.read_csv(filepath_or_buffer=DATASET_PATH)
        self.gdf = gpd.GeoDataFrame(
            self.df,
            geometry=gpd.points_from_xy(
                self.df.pickup_longitude, self.df.pickup_latitude
            ),
            crs=WGS_84,
        )

        # Convert to Web Mercator for plotting
        self.gdf.to_crs(WEB_MERCATOR, inplace=True)

    def set_query_bounds(self, is_start: bool, loc: Point, radius: float):
        if is_start:
            # A one row gdf containing a buffer around the point
            self.gdf_start = gpd.GeoDataFrame(
                data={"geometry": [loc]}, geometry="geometry", crs=WGS_84
            )
            self.gdf_start.to_crs(WEB_MERCATOR, inplace=True)
            self.gdf_start["geometry"] = self.gdf_start["geometry"].buffer(radius)
            self.gdf_start.set_geometry("geometry", inplace=True)
        else:
            # A one row gdf containing a buffer around the point
            self.gdf_end = gpd.GeoDataFrame(
                data={"geometry": [loc]}, geometry="geometry", crs=WGS_84
            )
            self.gdf_end.to_crs(WEB_MERCATOR, inplace=True)
            self.gdf_end["geometry"] = self.gdf_end["geometry"].buffer(radius)
            self.gdf_end.set_geometry("geometry", inplace=True)

    def intersect_query_bounds(self) -> int:
        # Intersect the data with the start bounds
        result = self.gdf.overlay(self.gdf_start, how="intersection")

        # Reset the geometry to the dropoff points instead
        result["geometry"] = gpd.points_from_xy(
            result.dropoff_longitude, result.dropoff_latitude, crs=WGS_84
        )
        result.set_geometry("geometry", inplace=True)
        result.to_crs(WEB_MERCATOR, inplace=True)

        # Intersect the data with the end bounds
        result = result.overlay(self.gdf_end, how="intersection")

        return len(result.index)


class MapControls:
    taxi_querier: TaxiQuerier

    start_slider: Slider
    end_slider: Slider
    start_radius: float
    end_radius: float

    submit_button: Button

    query_result: int
    query_result_text: Text

    def __init__(self, fig: plt.Figure):
        self.taxi_querier = TaxiQuerier()
        self.start_radius = 0
        self.end_radius = 0
        self.query_result = 0

        self.setup_sliders(fig=fig)
        self.setup_submit(fig=fig)

    def _set_radius(self, radius: float, is_start: bool) -> None:
        if is_start:
            self.start_radius = radius
        else:
            self.end_radius = radius

    def setup_sliders(self, fig: plt.Figure):
        ax_start = fig.add_axes([0.8, 0.6, 0.15, 0.03])
        ax_end = fig.add_axes([0.8, 0.4, 0.15, 0.03])

        ax_start.set_title("Start Radius (km)")
        ax_end.set_title("End Radius (km)")

        self.start_slider = Slider(
            ax=ax_start,
            label="",
            valmin=0,
            valmax=1000,
            valinit=self.start_radius,
            valstep=1.0,
        )
        self.end_slider = Slider(
            ax=ax_end,
            label="",
            valmin=0,
            valmax=1000,
            valinit=self.end_radius,
            valstep=1.0,
        )

        self.start_slider.on_changed(lambda x: self._set_radius(x, is_start=True))
        self.end_slider.on_changed(lambda x: self._set_radius(x, is_start=False))

    def setup_submit(self, fig: plt.Figure):
        ax_submit = fig.add_axes([0.8, 0.2, 0.15, 0.03])
        ax_result_text = fig.add_axes([0.8, 0.15, 0.15, 0.03])
        ax_result_text.set_axis_off()

        self.submit_button = Button(ax=ax_submit, label="Submit")
        self.submit_button.on_clicked(lambda x: self.submit_query())

        self.query_result_text = ax_result_text.text(
            0.5, 0.5, f"Query Result: {self.query_result}", ha="center", va="center"
        )

    def submit_query(self):
        self.taxi_querier.set_query_bounds(
            is_start=True, loc=Point(-74.00, 40.75), radius=self.start_radius
        )
        self.taxi_querier.set_query_bounds(
            is_start=False, loc=Point(-73.98, 40.76), radius=self.end_radius
        )
        self.query_result = self.taxi_querier.intersect_query_bounds()
        self.query_result_text.set_text(f"Query Result: {self.query_result}")


class Map:
    fig: plt.Figure
    ax_map: plt.Axes

    nyc_map: NYCMap
    controls: MapControls

    def __init__(self):
        self.setup_map()

        self.nyc_map = NYCMap(self.fig, self.ax_map)
        self.controls = MapControls(self.fig)

        plt.show()

    def setup_map(self):
        """
        Configure and setup the plotting of the map.
        """

        # Create the figure and axes for the map
        self.fig, self.ax_map = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(right=0.75)

        # Configure the plotted graphs information for display
        self.fig.suptitle(f"Taxicab Transit Information", fontsize=12)
        self.ax_map.set_xlabel("Longitude", fontsize=10)
        self.ax_map.set_ylabel("Latitude", fontsize=10)


if __name__ == "__main__":
    Map()
    # df = pd.read_csv(filepath_or_buffer=PATH)

    # # Define the gdf for the entire dataset, taking the geometry as the pickup points
    # gdf = gpd.GeoDataFrame(
    #     df,
    #     geometry=gpd.points_from_xy(df.pickup_longitude, df.pickup_latitude),
    #     crs="EPSG:4326",
    # )
    # gdf = gdf.to_crs("EPSG:3857")

    # # Form the gdf for the starting boundary about the point
    # print("Define the starting area of your query")
    # start = get_area()
    # start_gdf = gpd.GeoDataFrame(
    #     data={"geometry": [start[0]]},
    #     geometry="geometry",
    #     crs="EPSG:4326",
    # )
    # start_gdf.to_crs("EPSG:3857", inplace=True)
    # start_gdf["geometry"] = start_gdf["geometry"].buffer(start[1])
    # start_gdf.set_geometry("geometry", inplace=True)

    # # Form the gdf for the ending boundary about the point
    # print("Define the ending area of your query")
    # end = get_area()
    # end_gdf = gpd.GeoDataFrame(
    #     data={"geometry": [end[0]]}, geometry="geometry", crs="EPSG:4326"
    # )
    # end_gdf.to_crs("EPSG:3857", inplace=True)
    # end_gdf["geometry"] = end_gdf["geometry"].buffer(end[1])
    # end_gdf.set_geometry("geometry", inplace=True)

    # # Intersect the two gdfs and print the result
    # gdf = gdf.overlay(start_gdf, how="intersection")

    # # Reset the geometry to the dropoff points instead
    # gdf["geometry"] = gpd.points_from_xy(
    #     gdf.dropoff_longitude, gdf.dropoff_latitude, crs="EPSG:4326"
    # )
    # gdf.set_geometry("geometry", inplace=True)
    # gdf.to_crs("EPSG:3857", inplace=True)

    # # Intersect the two gdfs and print the result
    # gdf = gdf.overlay(end_gdf, how="intersection")

    # # Display the New York Boroughs
    # boroughs = gpd.read_file(gpd.datasets.get_path("nybb"))
    # boroughs.to_crs("EPSG:3857", inplace=True)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # boroughs.plot(ax=ax, alpha=0.4, color="grey")
    # cx.add_basemap(ax)

    # # Plot the start and end points
    # gpd.GeoDataFrame(
    #     data={"geometry": [start[0]]},
    #     geometry="geometry",
    #     crs="EPSG:4326",
    # ).plot(ax=ax, color="red", alpha=0.5)
    # gpd.GeoDataFrame(
    #     data={"geometry": [end[0]]}, geometry="geometry", crs="EPSG:4326"
    # ).plot(ax=ax, color="blue", alpha=0.5)

    # # Configure the plotted graphs information for display
    # fig.suptitle(f"Taxicab Transit Information: {len(gdf.index)} Trips", fontsize=12)
    # ax.set_xlabel("Longitude", fontsize=10)
    # ax.set_ylabel("Latitude", fontsize=10)
    # ax.set_xlim(min(start[0].x, end[0].x) - 0.1, max(start[0].x, end[0].x) + 0.1)
    # ax.set_ylim(min(start[0].y, end[0].y) - 0.1, max(start[0].y, end[0].y) + 0.1)
    # plt.show()

    # print(gdf)
