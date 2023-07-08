import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Dict, List

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from shapely.geometry import Point  # type: ignore

from adapted_sampling_method import AdapatedSamplingMethod

DATASET_PATH = os.path.join(Path(__file__).parent.parent, "AIS_2019_01_01.csv")
EXCESS_COLUMNS = [
    "SOG",
    "COG",
    "Heading",
    "IMO",
    "Status",
    "Length",
    "Width",
    "Draft",
    "Cargo",
    "CallSign",
    "VesselType",
    "TransceiverClass",
]

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude


class Privatiser:
    ship_id: str
    method: AdapatedSamplingMethod
    gdf: gpd.GeoDataFrame

    def __init__(self, method: AdapatedSamplingMethod) -> None:
        self.method = method
        self.gdf = self.process_initial_data()

    @classmethod
    def calculate_euclidean_error(
        cls, gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame
    ) -> float:
        """
        Calculates the Euclidean distance between two points
        """
        error = 0.0

        for i in range(len(gdf1)):
            point1 = gdf1.geometry.iloc[i]
            point2 = gdf2.geometry.iloc[i]
            error += np.linalg.norm(
                np.array([point1.x - point2.x, point1.y - point2.y])
            )

        return error

    def process_initial_data(self) -> gpd.GeoDataFrame:
        """
        Process the initial data according to a given input ship id
        """

        # Testing done primarily on ship 368048550
        self.ship_id = input("Enter a id to plot trajectory: ")

        df = pd.read_csv(DATASET_PATH)
        df = df.loc[df["MMSI"] == int(self.ship_id)]
        df.drop(
            labels=EXCESS_COLUMNS,
            axis=1,
            inplace=True,
        )

        # Convert to GeoDataFrame with web mercator projection of lat, lon
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LON, df.LAT),
            crs=WGS_84,
        )

        return df.to_crs(WEB_MERCATOR)

    def privatise_trajectory(self, eps: float) -> gpd.GeoDataFrame:
        """
        Privatises the trajectory of a ship using the exponential mechanism
        from the given method.
        """

        # Privatised route starts and ends at the same location as the real route
        privatised = {
            "geometry": [self.gdf.geometry.iloc[0]],
        }

        for i in range(1, len(self.gdf) - 1):
            # The current (real) point and the last (privatised) point
            current_point: Point = self.gdf.geometry.iloc[i]
            last_estimate: Point = privatised["geometry"][-1]

            # Get the distance and angle from the vector between the two points
            v = np.array(
                [
                    current_point.x - last_estimate.x,
                    current_point.y - last_estimate.y,
                ]
            )
            distance = np.linalg.norm(v)
            angle = (np.arctan2(v[1], v[0]) + 2 * np.pi) % (2 * np.pi)

            # Sample a distance and angle from the given distributions
            sampled_distance = self.method.sample_distance(eps, distance)
            sampled_angle = self.method.sample_angle(eps, angle)

            # Calculate the new point at angle sampled_angle in a circle of radius sampled_distance
            privatised["geometry"].append(
                Point(
                    current_point.x + sampled_distance * np.cos(sampled_angle),
                    current_point.y + sampled_distance * np.sin(sampled_angle),
                )
            )

        privatised["geometry"].append(self.gdf.geometry.iloc[-1])
        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)

    def generate_mean_max_errors(self) -> Dict[str, List[float]]:
        """
        Generates the mean and max errors for the given method
        We run 5 tests with values of epsilon [0.01, 0.1, 1, 10, 100]
        Each test is run 500 times and the mean and max errors are calculated
        """
        result: Dict[str, List[float]] = {"eps": [], "mean": [], "max": []}

        for eps in [0.01, 0.1, 1, 10, 100]:
            total_error = 0.0
            max_error = 0.0
            iterations = 1000

            for _ in range(iterations):
                privatised_df = self.privatise_trajectory(eps)
                error = self.calculate_euclidean_error(self.gdf, privatised_df)
                total_error += error
                max_error = max(max_error, error)

            mean = total_error / iterations

            result["eps"].append(eps)
            result["mean"].append(mean)
            result["max"].append(max_error)

        return result


if __name__ == "__main__":
    privatiser = Privatiser(AdapatedSamplingMethod())

    ship_id = privatiser.ship_id
    df = privatiser.gdf
    privatised_df = privatiser.privatise_trajectory(0.01)

    # res = privatiser.generate_mean_max_errors()
    # plt.plot(
    #     res["eps"],
    #     res["mean"],
    #     label="Mean",
    #     linewidth=1,
    #     linestyle="-",
    #     color="red",
    #     marker="o",
    #     markersize=5,
    #     markerfacecolor="red",
    # )
    # plt.plot(
    #     res["eps"],
    #     res["max"],
    #     label="Max",
    #     linewidth=1,
    #     linestyle="-",
    #     color="blue",
    #     marker="o",
    #     markersize=5,
    #     markerfacecolor="blue",
    # )
    # plt.xscale("log")
    # plt.show()

    # Plot the trajectory and the start and end points
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Trajectory of ship {ship_id}")

    plt.plot(
        df.geometry.x,
        df.geometry.y,
        linewidth=1,
        linestyle="--",
        color="green",
        marker="o",
        markersize=5,
        markerfacecolor="red",
    )
    plt.plot(
        privatised_df.geometry.x,
        privatised_df.geometry.y,
        linewidth=1,
        linestyle="--",
        color="blue",
        marker="o",
        markersize=5,
        markerfacecolor="orange",
    )

    plt.annotate("Start", (df.geometry.iloc[0].x, df.geometry.iloc[0].y))
    plt.annotate("End", (df.geometry.iloc[-1].x, df.geometry.iloc[-1].y))
    plt.axis("square")
    plt.ticklabel_format(useOffset=False)

    plt.show()
