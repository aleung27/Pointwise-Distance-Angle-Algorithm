import os
from pathlib import Path

import contextily
import geopandas as gpd
import geopy.geocoders.nominatim as nominatim
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.text import Text
from matplotlib.widgets import Button, Slider, TextBox
from shapely.geometry import Point

DATASET_PATH = os.path.join(Path(__file__).parent.parent, "train.csv")
NEW_YORK_TIF = "new_york.tif"

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude


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
            print("Downloading New York map (this can take a while)...")
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
    """
    Class which controls the overall querying of the taxi data.
    Contains the underlying data from the taxicab dataset as a geodataframe.
    The start and end bounds are stored as geodataframes which can be intersected
    to determine the number of trips which fit the query.
    """

    df: pd.DataFrame  # The underlying dataframe for the taxi data
    gdf: gpd.GeoDataFrame  # The transformed df for the taxi data with lat/long geometry column

    gdf_start: gpd.GeoDataFrame  # The gdf for the starting boundary about the point
    gdf_end: gpd.GeoDataFrame  # The gdf for the ending boundary about the point

    def __init__(self):
        self.load_data()

        self.gdf_start = None
        self.gdf_end = None

    def load_data(self):
        """
        Load the taxi data from the csv file into a dataframe and transform it into a geodataframe.
        Sets the geometry column to be initially be the pickup points.
        Geometry projection is initially in WGS_84 (lat/long) and is converted to Web Mercator for plotting.
        """

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

    def set_start_query_bounds(self, loc: Point, radius: float):
        """
        Set the query bounds for the start location. This is a one row gdf
        which contains a singular point and a buffer around it. The point is
        initially in WGS_84 and is converted to Web Mercator to apply the buffer
        """

        # A one row gdf containing a buffer around the point
        self.gdf_start = gpd.GeoDataFrame(
            data={"geometry": [loc]}, geometry="geometry", crs=WGS_84
        )
        self.gdf_start.to_crs(WEB_MERCATOR, inplace=True)
        self.gdf_start["geometry"] = self.gdf_start["geometry"].buffer(radius)
        self.gdf_start.set_geometry("geometry", inplace=True)

    def set_end_query_bounds(self, loc: Point, radius: float):
        """
        Set the query bounds for the end location. This is a one row gdf
        which contains a singular point and a buffer around it. The point is
        initially in WGS_84 and is converted to Web Mercator to apply the buffer
        """

        # A one row gdf containing a buffer around the point
        self.gdf_end = gpd.GeoDataFrame(
            data={"geometry": [loc]}, geometry="geometry", crs=WGS_84
        )
        self.gdf_end.to_crs(WEB_MERCATOR, inplace=True)
        self.gdf_end["geometry"] = self.gdf_end["geometry"].buffer(radius)
        self.gdf_end.set_geometry("geometry", inplace=True)

    def intersect_query_bounds(self) -> int:
        """
        Perform the query against the dataset, intersecting the dataset with
        the provided start and end bounds to determine which taxis started within
        the start buffer and ended within the end buffer.
        """

        # Intersect the departures with the start bounds
        result = self.gdf.overlay(self.gdf_start, how="intersection")

        # Reset the geometry to the dropoff points instead
        result["geometry"] = gpd.points_from_xy(
            result.dropoff_longitude, result.dropoff_latitude, crs=WGS_84
        )
        result.set_geometry("geometry", inplace=True)
        result.to_crs(WEB_MERCATOR, inplace=True)

        # Intersect the arrivals with the end bounds
        result = result.overlay(self.gdf_end, how="intersection")

        return len(result.index)

    def plot_query_bounds(self, ax: plt.Axes):
        """
        Plot the start and end bounds onto the provided Axes object.
        Removes any previously plotted bounds which should be the last
        2 objects plotted to the axes.
        """

        # Remove any plotted start/end bounds
        while len(ax.collections) != 1:
            ax.collections[-1].remove()

        # Plot the start and end bounds
        self.gdf_start.to_crs(WGS_84).plot(ax=ax, color="blue", alpha=0.5)
        self.gdf_end.to_crs(WGS_84).plot(ax=ax, color="red", alpha=0.5)


class MapControls:
    """
    Controls the querier and widgets used to selected the start and end points.
    Has a reference to a TaxiQuerier instance to query the data.
    """

    taxi_querier: TaxiQuerier

    # Widgets for the start point
    start_slider: Slider
    start_coords_box: TextBox

    # Stores the values for the start settings
    start_coords: str
    start_radius: float

    # Widgets for the end point
    end_slider: Slider
    end_coords_box: TextBox

    # Stores the values for the end settings
    end_coords: str
    end_radius: float

    # Widgets for the submission and display of query
    submit_button: Button
    query_result_text: Text
    query_result: int

    def __init__(self, fig: plt.Figure, ax: plt.Axes):
        self.taxi_querier = TaxiQuerier()

        # Set a default 1km circle around some default point in Manhattan
        self.start_radius = 1.0
        self.end_radius = 1.0
        self.query_result = 0

        self.start_coords = "-74.0,40.74"
        self.end_coords = "-73.98,40.76"

        self.setup_start_controls(fig=fig)
        self.setup_end_controls(fig=fig)
        self.setup_query_submit(fig=fig, ax=ax)

        # Load the map with some data
        self.submit_query(ax=ax)

    def setup_start_controls(self, fig: plt.Figure):
        """
        Sets up the widgets for the start point selection.
        Contains a textbox for the coordinates and a slider for the radius.
        The textbox takes in lat,lon coordinates as a space-separated pair
        The slider taken in a radius in km's between 0-10
        """

        ax_start_textbox = fig.add_axes([0.8, 0.7, 0.15, 0.03])
        ax_start_slider = fig.add_axes([0.8, 0.65, 0.15, 0.03])

        ax_start_textbox.set_title("Start Point Selection")

        self.start_coords_box = TextBox(
            ax=ax_start_textbox,
            label="Coordinates (lon, lat)",
            textalignment="left",
            initial=self.start_coords,
        )
        self.start_slider = Slider(
            ax=ax_start_slider,
            label="Radius (km)",
            valmin=0,
            valmax=10,
            valinit=self.start_radius,
            valstep=0.1,
        )

        self.start_coords_box.on_submit(lambda x: self.__setattr__("start_coords", x))
        self.start_slider.on_changed(lambda x: self.__setattr__("start_radius", x))

    def setup_end_controls(self, fig: plt.Figure):
        """
        Sets up the widgets for the end point selection.
        Contains a textbox for the coordinates and a slider for the radius.
        The textbox takes in lat,lon coordinates as a space-separated pair
        The slider taken in a radius in km's between 0-10
        """

        ax_end_textbox = fig.add_axes([0.8, 0.4, 0.15, 0.03])
        ax_end_slider = fig.add_axes([0.8, 0.35, 0.15, 0.03])

        ax_end_textbox.set_title("End Point Selection")

        self.end_coords_box = TextBox(
            ax=ax_end_textbox,
            label="Coordiantes (lon, lat)",
            textalignment="left",
            initial=self.end_coords,
        )
        self.end_slider = Slider(
            ax=ax_end_slider,
            label="Radius (km)",
            valmin=0,
            valmax=10,
            valinit=self.end_radius,
            valstep=0.1,
        )

        self.end_coords_box.on_submit(lambda x: self.__setattr__("end_coords", x))
        self.end_slider.on_changed(lambda x: self.__setattr__("end_radius", x))

    def setup_query_submit(self, fig: plt.Figure, ax: plt.Axes):
        """
        Setup the widgets for submitting the query and displaying the result.
        Contains a submission button which will submit the query and update the result text.
        """

        ax_submit_button = fig.add_axes([0.8, 0.2, 0.15, 0.03])
        ax_result_text = fig.add_axes([0.8, 0.15, 0.15, 0.03])

        ax_result_text.set_axis_off()
        self.query_result_text = ax_result_text.text(
            0.5, 0.5, f"Query Result: {self.query_result}", ha="center", va="center"
        )

        self.submit_button = Button(ax=ax_submit_button, label="Submit Query")
        self.submit_button.on_clicked(lambda _: self.submit_query(ax=ax))

    def submit_query(self, ax: plt.Axes):
        """
        Performs a query against the TaxiQuerier instance and updates the result text.
        First updates the query bounds for both the start and end points before
        performing an intersection with the dataset and plotting the resultant bounds.
        """

        # Set the new query bounds for the start and end
        self.taxi_querier.set_start_query_bounds(
            loc=Point(
                float(self.start_coords.split(",")[0]),
                float(self.start_coords.split(",")[1]),
            ),
            radius=self.start_radius * 1000,
        )
        self.taxi_querier.set_end_query_bounds(
            loc=Point(
                float(self.end_coords.split(",")[0]),
                float(self.end_coords.split(",")[1]),
            ),
            radius=self.end_radius * 1000,
        )

        # Perform the query and update the result text
        self.query_result = self.taxi_querier.intersect_query_bounds()
        self.query_result_text.set_text(f"Query Result: {self.query_result}")

        # Plot the query bounds
        self.taxi_querier.plot_query_bounds(ax=ax)


class Map:
    """
    Controls the overall displayed map, containing the NYCMap and MapControls.
    Also controls the overarching figure and main axes for the map.
    """

    fig: plt.Figure
    ax_map: plt.Axes

    nyc_map: NYCMap
    controls: MapControls

    def __init__(self):
        self.setup_map()

        self.nyc_map = NYCMap(self.fig, self.ax_map)
        self.controls = MapControls(self.fig, self.ax_map)

        plt.show()

    def setup_map(self):
        """
        Configure and setup the plotting of the map.
        """

        # Create the figure and axes for the map
        self.fig, self.ax_map = plt.subplots(figsize=(16, 9))
        self.fig.subplots_adjust(right=0.75)

        # Configure the plotted graphs information for display
        self.fig.suptitle(f"Taxicab Transit Information", fontsize=12)
        self.ax_map.set_xlabel("Longitude", fontsize=10)
        self.ax_map.set_ylabel("Latitude", fontsize=10)


if __name__ == "__main__":
    Map()
