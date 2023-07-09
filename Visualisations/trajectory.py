import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Dict, List

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from shapely.geometry import Point  # type: ignore

from sample_method import SampleMethod
from adapted_sampling_method import AdapatedSamplingMethod
from sdd import SampleDistanceDirection

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
    gdf: gpd.GeoDataFrame

    def __init__(self) -> None:
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

    def generate_mean_max_errors(self, methods: List[SampleMethod]) -> pd.DataFrame:
        """
        Generates the mean and max errors for the given method
        We run 5 tests with values of epsilon [0.01, 0.1, 1, 10, 100]
        Each test is run 1000 times and the mean and max errors are calculated
        """
        result: Dict[str, List[float | str]] = {
            "method": [],
            "eps": [],
            "mean": [],
            "max": [],
        }
        iterations = 500

        for m in methods:
            for eps in [0.01, 0.1, 1, 10, 100]:
                total_error = 0.0
                max_error = 0.0

                for _ in range(iterations):
                    privatised_df = m.privatise_trajectory(self.gdf, eps)
                    error = self.calculate_euclidean_error(self.gdf, privatised_df)

                    total_error += error
                    max_error = max(max_error, error)

                mean = total_error / iterations

                result["method"].append(m.__class__.__name__)
                result["eps"].append(eps)
                result["mean"].append(mean)
                result["max"].append(max_error)

        return pd.DataFrame.from_dict(result)


if __name__ == "__main__":
    privatiser = Privatiser()

    ship_id = privatiser.ship_id
    df = privatiser.gdf

    sdd_gdf = SampleDistanceDirection().privatise_trajectory(df, 1)
    asm_gdf = AdapatedSamplingMethod().privatise_trajectory(df, 1)

    res = privatiser.generate_mean_max_errors(
        methods=[SampleDistanceDirection(), AdapatedSamplingMethod()]
    )
    plt.plot(
        res[res["method"] == SampleDistanceDirection.__name__]["eps"],
        res[res["method"] == SampleDistanceDirection.__name__]["mean"],
        label="Mean (SDD)",
        linewidth=1,
        linestyle="-",
        color="red",
        marker="o",
        markersize=5,
        markerfacecolor="red",
    )
    plt.plot(
        res[res["method"] == SampleDistanceDirection.__name__]["eps"],
        res[res["method"] == SampleDistanceDirection.__name__]["max"],
        label="Max (SDD)",
        linewidth=1,
        linestyle="--",
        color="red",
        marker="o",
        markersize=5,
        markerfacecolor="red",
    )
    plt.plot(
        res[res["method"] == AdapatedSamplingMethod.__name__]["eps"],
        res[res["method"] == AdapatedSamplingMethod.__name__]["mean"],
        label="Max",
        linewidth=1,
        linestyle="-",
        color="blue",
        marker="o",
        markersize=5,
        markerfacecolor="blue",
    )
    plt.plot(
        res[res["method"] == AdapatedSamplingMethod.__name__]["eps"],
        res[res["method"] == AdapatedSamplingMethod.__name__]["max"],
        label="Max",
        linewidth=1,
        linestyle="--",
        color="blue",
        marker="o",
        markersize=5,
        markerfacecolor="blue",
    )

    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Euclidean Pointwise Error (m)")
    plt.title(f"Trajectory of ship {ship_id}")
    plt.xscale("log")
    plt.legend()

    plt.show()

    # Plot the trajectory and the start and end points
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.title(f"Trajectory of ship {ship_id}")

    # plt.plot(
    #     df.geometry.x,
    #     df.geometry.y,
    #     linewidth=1,
    #     linestyle="--",
    #     color="green",
    #     marker="o",
    #     markersize=5,
    #     markerfacecolor="red",
    # )
    # plt.plot(
    #     asm_gdf.geometry.x,
    #     asm_gdf.geometry.y,
    #     linewidth=1,
    #     linestyle="--",
    #     color="blue",
    #     marker="o",
    #     markersize=5,
    #     markerfacecolor="orange",
    # )
    # plt.plot(
    #     sdd_gdf.geometry.x,
    #     sdd_gdf.geometry.y,
    #     linewidth=1,
    #     linestyle="--",
    #     color="yellow",
    #     marker="o",
    #     markersize=5,
    #     markerfacecolor="yellow",
    # )

    # plt.annotate("Start", (df.geometry.iloc[0].x, df.geometry.iloc[0].y))
    # plt.annotate("End", (df.geometry.iloc[-1].x, df.geometry.iloc[-1].y))
    # plt.axis("square")
    # plt.ticklabel_format(useOffset=False)

    # plt.show()
