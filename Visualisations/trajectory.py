import pandas as pd
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

DATASET_FOLDER = os.path.join(Path(__file__).parent.parent, "AIS_2019_01_01.csv")

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude


if __name__ == "__main__":
    ship_id = input("Enter a id to plot trajectory: ")
    df = pd.read_csv(DATASET_FOLDER)

    # Filter the dataframe to only include the ship with the given id
    ship_id_df = df[df["MMSI"] == int(ship_id)]
    ship_id_gdf = gpd.GeoDataFrame(
        ship_id_df,
        geometry=gpd.points_from_xy(ship_id_df.LON, ship_id_df.LAT),
    )

    # Plot the trajectory and the start and end points
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory of ship {ship_id}")
    plt.plot(
        ship_id_gdf.LON,
        ship_id_gdf.LAT,
        linewidth=1,
        linestyle="--",
        color="green",
        marker="o",
        markersize=5,
        markerfacecolor="red",
    )
    plt.annotate("Start", (ship_id_gdf.LON.iloc[0], ship_id_gdf.LAT.iloc[0]))
    plt.annotate("End", (ship_id_gdf.LON.iloc[-1], ship_id_gdf.LAT.iloc[-1]))
    plt.show()
