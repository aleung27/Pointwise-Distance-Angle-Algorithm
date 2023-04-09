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

PATH = os.path.join(Path(__file__).parent.parent, "train.csv")
NEW_YORK_TIF = "new_york.tif"
WEB_MERCATOR = "EPSG:3857"


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
        boroughs.to_crs(WEB_MERCATOR, inplace=True)

        # Plot NY and add the OSM base map for the overall display
        boroughs.plot(ax=ax, color="none", linewidth=0)
        contextily.add_basemap(ax=ax, source=NEW_YORK_TIF)


class Map:
    fig: plt.Figure
    ax: plt.Axes
    nyc_map: NYCMap

    def __init__(self):
        self.setup_map()
        self.nyc_map = NYCMap(self.fig, self.ax)

        plt.show()

    def setup_map(self):
        """
        Configure and setup the plotting of the map.
        """

        # Create the figure and axes for the map
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Configure the plotted graphs information for display
        self.fig.suptitle(f"Taxicab Transit Information", fontsize=12)
        self.ax.set_xlabel("Longitude", fontsize=10)
        self.ax.set_ylabel("Latitude", fontsize=10)


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
