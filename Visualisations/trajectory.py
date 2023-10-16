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
from avoidant_sampling_method import AvoidantSamplingMethod
from constants import WEB_MERCATOR, WGS_84

DATASET_PATH = os.path.join(Path(__file__).parent.parent, "AIS_2019_01_01.csv")
FLORIDA_COASTLINE = "Shapefiles/florida_coastline.shp"

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

MODES = {
    "q": "Quit",
    "p": "Privatise",
    "e": "Error Comparison",
    "r": "Region Testing Visualisation (Avoidant Sampling Method)",
}
TESTING_EPSILONS = [0.01, 0.1, 1, 5, 10, 100]


class Privatiser:
    gdf: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame

    def __init__(self) -> None:
        self.gdf = self.load_initial_data()
        self.florida_coastline = self.load_florida_coastline()

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
            error += float(
                np.linalg.norm(np.array([point1.x - point2.x, point1.y - point2.y]))
            )

        return error

    def load_initial_data(self) -> gpd.GeoDataFrame:
        df = pd.read_csv(DATASET_PATH)
        df.drop(
            labels=EXCESS_COLUMNS,
            axis=1,
            inplace=True,
        )

        # Convert to GeoDataFrame with web mercator projection of lat, lon
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LON, df.LAT),
            crs=WGS_84,
        )

        return gdf.to_crs(WEB_MERCATOR)

    def load_florida_coastline(self) -> gpd.GeoDataFrame:
        florida_coastline = gpd.read_file(FLORIDA_COASTLINE)

        return florida_coastline.to_crs(WEB_MERCATOR)

    def run_command(self):
        commands = "\n".join([f"\t- {v} ({k})" for k, v in MODES.items()])
        mode = input(f"Enter a command:\n {commands}\n")

        if mode == "q":
            exit()
        elif mode == "p":
            self.command_privatise()
        elif mode == "r":
            self.command_region_testing()
        elif mode == "e":
            self.command_error_plot()
        else:
            print("Invalid command")

    def command_error_plot(self):
        """
        Generates the mean and max errors for each of the given methods
        for different values of epsilon.

        Results are plotted on a graph.
        """

        N = 500  # Number of iterations to run for each epsilon value
        methods: List[SampleMethod] = [
            SampleDistanceDirection(),
            AdapatedSamplingMethod(),
            AvoidantSamplingMethod(self.florida_coastline),
        ]

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()

        # For each method, run through each test value of epsilon N times
        for m in methods:
            res: Dict[str, List[float]] = {
                "eps": [],
                "mean": [],
                "max": [],
            }

            for eps in TESTING_EPSILONS:
                total_error = 0.0
                max_error = 0.0

                for _ in range(N):
                    privatised_df = m.privatise_trajectory(gdf, eps)
                    error = self.calculate_euclidean_error(gdf, privatised_df)

                    total_error += error
                    max_error = max(max_error, error)

                mean_error = total_error / N

                res["eps"].append(eps)
                res["mean"].append(mean_error)
                res["max"].append(max_error)

            # Plot the results for the given method
            plt.plot(
                res["eps"],
                res["mean"],
                label=f"Mean ({m.__class__.__name__})",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="o",
                markersize=5,
                markerfacecolor=m.COLOR,
            )
            plt.plot(
                res["eps"],
                res["max"],
                label=f"Max ({m.__class__.__name__})",
                linewidth=1,
                linestyle="--",
                color=m.COLOR,
                marker="o",
                markersize=5,
                markerfacecolor=m.COLOR,
            )

        # Configure the axes labels and the title of the plot
        plt.xlabel("Epsilon (ε)")
        plt.ylabel("Euclidean Pointwise Error (m)")
        plt.title(
            f"Mean and Max Euclidean Error for Different Privitisation Methods (ship {ship_id})"
        )
        plt.xscale("log")
        plt.legend()

        plt.show()

    def command_region_testing(self):
        """
        Ask the user to supply a given web mercator point and a radius.
        Perform a test for sampling angles for different values of epsilon
        and plot the results on a graph.

        Tested on: p = -9064783 3544363, d = 100
        Try on -9050513 3481824, d = 1000
        """
        N = 1  # Number of samples to take for each epsilon value

        p = input("Enter a space-separated xy coordinate pair: ").split(" ")
        p = Point(float(p[0]), float(p[1]))
        d = float(input("Enter a value for the radius to sample at: "))

        sampler = AvoidantSamplingMethod(self.florida_coastline)
        valid_regions, invalid_regions = sampler.find_restricted_regions(p, d)
        gdfs = []  # Store the resultant points for each epsilon test

        # For each epsilon value, sample N points and plot them
        for eps in TESTING_EPSILONS:
            points = []

            for _ in range(N):
                angle = sampler._sample_angle(eps, 0, valid_regions, invalid_regions)
                points.append(Point(p.x + d * np.cos(angle), p.y + d * np.sin(angle)))

            gdfs.append(gpd.GeoDataFrame(geometry=points, crs=WEB_MERCATOR))

        # Plot the given results in one figure with 6 Axes
        plt.cla()
        plt.clf()
        plt.close()

        # 2 rows of 3 columns with shared axes
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))

        for i in range(len(TESTING_EPSILONS)):
            ax: plt.Axes = axs[i // 3, i % 3]  # Current axis to plot on

            # Plot the points and circle onto the respective axis as well as the coastline
            gdfs[i].plot(ax=ax, color="red", alpha=0.1)
            ax.plot(p.x, p.y, marker="o", markersize=1, color="blue")
            gpd.GeoDataFrame(geometry=[p.buffer(d).boundary], crs=WEB_MERCATOR).plot(
                ax=ax, color="blue", alpha=0.5, linestyle="--"
            )
            self.florida_coastline.cx[
                p.x - 2 * d : p.x + 2 * d, p.y - 2 * d : p.y + 2 * d
            ].plot(ax=ax, color="black", alpha=0.5)

            # Configure the axes labels
            ax.set_title(f"ε = {TESTING_EPSILONS[i]}")
            ax.set_xlabel("Lon (Mercator)")
            ax.set_ylabel("Lat (Mercator)")

        for ax in axs.flat:
            ax.label_outer()

        # Configure the figure and layout of the subplots
        fig.suptitle(
            f"Obstacle Avoidance Sampling Method for Differing Values of Epsilon (N = {N})"
        )
        fig.tight_layout()
        plt.subplots_adjust(wspace=0)
        plt.show()

    def command_privatise(self):
        """
        Privatise a given ship's trajectory using the 3 different methods
        and plot the results on a graph.

        Florida (matanzas river) Trajectories:
            - 367597490
            - 367181030
            - 368057420

        Small open ocean trajectory:
            - 368048550
        """

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        epsilon = float(input("Enter a value for epsilon (ε): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]

        # Run the privitisation against the 3 current schemes available
        # sdd_gdf = SampleDistanceDirection().privatise_trajectory(gdf, eps=epsilon)
        asm_gdf = AdapatedSamplingMethod().privatise_trajectory(gdf, eps=epsilon)
        avoid_gdf = AvoidantSamplingMethod(self.florida_coastline).privatise_trajectory(
            gdf, eps=epsilon
        )
        postprocessed_avoid_gdf = AvoidantSamplingMethod(
            self.florida_coastline
        ).postprocess_result(avoid_gdf, 200)

        # Reduce the polygons we plot by forming a bbox of the min & max lat and long
        gdfs = [gdf, asm_gdf, avoid_gdf]
        min_lon, min_lat, max_lon, max_lat = (
            np.min([df.geometry.x.min() for df in gdfs]),
            np.min([df.geometry.y.min() for df in gdfs]),
            np.max([df.geometry.x.max() for df in gdfs]),
            np.max([df.geometry.y.max() for df in gdfs]),
        )

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()

        # Set the axes labels and the title of the plot
        plt.xlabel("Lon (Mercator)")
        plt.ylabel("Lat (Mercator)")
        plt.title(f"Trajectory of Ship {ship_id} Under Different Methods")

        # Plot each of the trajectories
        plt.plot(
            gdf.geometry.x,
            gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="green",
            marker="o",
            markersize=5,
            markerfacecolor="green",
            label="Real Trajectory",
        )
        # plt.plot(
        #     sdd_gdf.geometry.x,
        #     sdd_gdf.geometry.y,
        #     linewidth=1,
        #     linestyle="--",
        #     color="purple",
        #     marker="o",
        #     markersize=5,
        #     markerfacecolor="purple",
        #     label="Sample Distance Direction Method",
        # )
        plt.plot(
            asm_gdf.geometry.x,
            asm_gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="blue",
            marker="o",
            markersize=5,
            markerfacecolor="blue",
            label="Adapted Sampling Method",
        )
        plt.plot(
            avoid_gdf.geometry.x,
            avoid_gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="orange",
            marker="o",
            markersize=5,
            markerfacecolor="orange",
            label="Avoidant Sampling Method",
        )
        plt.plot(
            postprocessed_avoid_gdf.geometry.x,
            postprocessed_avoid_gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="indianred",
            marker="o",
            markersize=5,
            markerfacecolor="indianred",
            label="Postprocessed Avoidant Sampling Method",
        )
        self.florida_coastline.cx[min_lon:max_lon, min_lat:max_lat].plot(
            ax=plt.gca(), color="black", alpha=0.5
        )

        # Annotate the start and end points of the trajectory
        plt.annotate("Start", (gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y))
        plt.annotate("End", (gdf.geometry.iloc[-1].x, gdf.geometry.iloc[-1].y))

        # Configure the display of the plot and show
        plt.axis("square")
        plt.ticklabel_format(useOffset=False)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    privatiser = Privatiser()

    while True:
        privatiser.run_command()
