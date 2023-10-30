import pandas as pd
import os
import numpy as np
from pathlib import Path
from typing import Dict, List

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from shapely.geometry import Point  # type: ignore
from matplotlib.ticker import StrMethodFormatter, NullFormatter, ScalarFormatter  # type: ignore

from sample_method import SampleMethod
from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection

from dtw import dtw, DTW  # type: ignore
from similaritymeasures import frechet_dist  # type: ignore
from math import inf

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
    "pd": "Privatise, varying delta",
    "pe": "Privatise, varying epsilon",
    "pp": "Privatise, preset",
    "e": "Error Comparison",
    "ep": "Error Comparison, preset",
    "ee": "Error Comparison, varying epsilon",
    "ed": "Error Comparison, varying delta",
}

PRESET_MMSIS = [
    258288000,
    316038559,
    367637340,
    368011140,
]
PRESET_EPSILONS = [0.1, 0.5, 1, 2, 5, 10]
PRESET_DELTAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
TEST_ITERATIONS = 100


class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%#.4g"


class Privatiser:
    gdf: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame

    def __init__(self) -> None:
        self.gdf = self.load_initial_data()
        self.florida_coastline = self.load_florida_coastline()

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
        elif mode == "pd":
            self.command_privatise_varying_delta()
        elif mode == "pe":
            self.command_privatise_varying_epsilon()
        elif mode == "pp":
            self.command_privatise_preset()
        elif mode == "e":
            self.command_error_preset()
        elif mode == "ee":
            self.command_error_varying_epsilon()
        elif mode == "ed":
            self.command_error_varying_delta()
        else:
            print("Invalid command")

    def command_error_varying_epsilon(self):
        """
        Generates the mean and max errors for each of the given methods
        for different values of epsilon.

        Results are plotted on a graph.
        """

        methods: List[SampleMethod] = [
            SampleDistanceDirection(),
            PointwiseDistanceAngle(self.florida_coastline),
        ]

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        delta = float(input("Enter a value for delta (δ): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]
        gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(14, 10))
        dtw_ax: plt.Axes = axs[0]
        dfd_ax: plt.Axes = axs[1]

        # For each method, run through each test value of epsilon N times
        for m in methods:
            res: Dict[str, List[float]] = {"eps": [], "dtw": [], "dfd": []}

            for eps in PRESET_EPSILONS:
                dtw_min, dtw_max, dtw_avg = inf, -inf, 0.0
                dfd_min, dfd_max, dfd_avg = inf, -inf, 0.0

                for _ in range(TEST_ITERATIONS):
                    privatised_df = m.privatise_trajectory(gdf, eps, delta)
                    privatised_pts = [
                        (pt.x, pt.y) for pt in privatised_df.geometry.values
                    ]

                    dfd_error = frechet_dist(privatised_pts, gdf_pts)
                    dtw_error = dtw(privatised_pts, gdf_pts).normalizedDistance

                    dtw_min = min(dtw_min, dtw_error)
                    dtw_max = max(dtw_max, dtw_error)
                    dtw_avg += dtw_error

                    dfd_min = min(dfd_min, dfd_error)
                    dfd_max = max(dfd_max, dfd_error)
                    dfd_avg += dfd_error

                dtw_avg /= TEST_ITERATIONS
                dfd_avg /= TEST_ITERATIONS

                res["eps"].append(eps)
                res["dfd"].append((dfd_avg, dfd_min, dfd_max))
                res["dtw"].append((dtw_avg, dtw_min, dtw_max))

            # Plot the results for the given method
            dtw_ax.errorbar(
                res["eps"],
                [y[0] for y in res["dtw"]],
                yerr=(
                    np.subtract([y[0] for y in res["dtw"]], [y[1] for y in res["dtw"]]),
                    np.subtract([y[2] for y in res["dtw"]], [y[0] for y in res["dtw"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )
            dfd_ax.errorbar(
                res["eps"],
                [y[0] for y in res["dfd"]],
                yerr=(
                    np.subtract([y[0] for y in res["dfd"]], [y[1] for y in res["dfd"]]),
                    np.subtract([y[2] for y in res["dfd"]], [y[0] for y in res["dfd"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )

        # Configure the axes labels
        dtw_ax.set_title(f"Normalised Dynamic Time Warping Error", y=0.9)
        dtw_ax.set_xlabel("Epsilon (ε)")
        dtw_ax.set_ylabel("Error (m)")
        dtw_ax.set_xscale("log")
        dtw_ax.set_xticks(PRESET_EPSILONS)
        dtw_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        dfd_ax.set_title(f"Discrete Fréchet Distance Error", y=0.9)
        dfd_ax.set_xlabel("Epsilon (ε)")
        dfd_ax.set_ylabel("Error (m)")
        dfd_ax.set_xscale("log")
        dfd_ax.set_xticks(PRESET_EPSILONS)
        dfd_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        # Configure the figure and layout of the subplots
        fig.suptitle(
            f"Trajectory Error Calculations for Varying Epsilon (N: {TEST_ITERATIONS}, δ: {delta})"
        )
        fig.legend(
            handles=dtw_ax.get_legend_handles_labels()[0],
            labels=dtw_ax.get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.subplots_adjust(wspace=0.2, right=0.85)

        plt.show()

    def command_error_varying_delta(self):
        """
        Generates the mean and max errors for each of the given methods
        for different values of delta.

        Results are plotted on a graph.
        """

        methods: List[SampleMethod] = [
            # SampleDistanceDirection(),
            PointwiseDistanceAngle(self.florida_coastline),
        ]

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        epsilon = float(input("Enter a value for epsilon (ε): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]
        gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(14, 10))
        dtw_ax: plt.Axes = axs[0]
        dfd_ax: plt.Axes = axs[1]

        # For each method, run through each test value of epsilon N times
        for m in methods:
            res: Dict[str, List[float]] = {"delta": [], "dtw": [], "dfd": []}

            for delta in PRESET_DELTAS:
                dtw_min, dtw_max, dtw_avg = inf, -inf, 0.0
                dfd_min, dfd_max, dfd_avg = inf, -inf, 0.0

                for _ in range(TEST_ITERATIONS):
                    privatised_df = m.privatise_trajectory(gdf, epsilon, delta)
                    privatised_pts = [
                        (pt.x, pt.y) for pt in privatised_df.geometry.values
                    ]

                    dfd_error = frechet_dist(privatised_pts, gdf_pts)
                    dtw_error = dtw(privatised_pts, gdf_pts).normalizedDistance

                    dtw_min = min(dtw_min, dtw_error)
                    dtw_max = max(dtw_max, dtw_error)
                    dtw_avg += dtw_error

                    dfd_min = min(dfd_min, dfd_error)
                    dfd_max = max(dfd_max, dfd_error)
                    dfd_avg += dfd_error

                dtw_avg /= TEST_ITERATIONS
                dfd_avg /= TEST_ITERATIONS

                res["delta"].append(delta)
                res["dfd"].append((dfd_avg, dfd_min, dfd_max))
                res["dtw"].append((dtw_avg, dtw_min, dtw_max))

            # Plot the results for the given method
            dtw_ax.errorbar(
                res["delta"],
                [y[0] for y in res["dtw"]],
                yerr=(
                    np.subtract([y[0] for y in res["dtw"]], [y[1] for y in res["dtw"]]),
                    np.subtract([y[2] for y in res["dtw"]], [y[0] for y in res["dtw"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )
            dfd_ax.errorbar(
                res["delta"],
                [y[0] for y in res["dfd"]],
                yerr=(
                    np.subtract([y[0] for y in res["dfd"]], [y[1] for y in res["dfd"]]),
                    np.subtract([y[2] for y in res["dfd"]], [y[0] for y in res["dfd"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )

        # Configure the axes labels
        dtw_ax.set_title(f"Normalised Dynamic Time Warping Error", y=0.9)
        dtw_ax.set_xlabel("Delta (δ)")
        dtw_ax.set_ylabel("Error (m)")
        dtw_ax.set_xscale("log")
        dtw_ax.set_xticks(PRESET_DELTAS)
        dtw_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        dfd_ax.set_title(f"Discrete Fréchet Distance Error", y=0.9)
        dfd_ax.set_xlabel("Delta (δ)")
        dfd_ax.set_ylabel("Error (m)")
        dfd_ax.set_xscale("log")
        dfd_ax.set_xticks(PRESET_DELTAS)
        dfd_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        # Configure the figure and layout of the subplots
        fig.suptitle(
            f"Trajectory Error Calculations for Ship {ship_id} Varying Delta (N: {TEST_ITERATIONS}, ε: {epsilon})"
        )
        fig.legend(
            handles=dtw_ax.get_legend_handles_labels()[0],
            labels=dtw_ax.get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.subplots_adjust(wspace=0.2, right=0.85)

        plt.show()

    def command_privatise_varying_delta(self):
        """
        Generates the mean and max errors for each of the given methods
        for different values of epsilon.

        Results are plotted on a graph.
        """

        methods: List[SampleMethod] = [
            SampleDistanceDirection(),
            PointwiseDistanceAngle(self.florida_coastline),
        ]

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        delta = float(input("Enter a value for delta (δ): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]
        gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(14, 10))
        dtw_ax: plt.Axes = axs[0]
        dfd_ax: plt.Axes = axs[1]

        # For each method, run through each test value of epsilon N times
        for m in methods:
            res: Dict[str, List[float]] = {"eps": [], "dtw": [], "dfd": []}

            for eps in PRESET_EPSILONS:
                dtw_min, dtw_max, dtw_avg = inf, -inf, 0.0
                dfd_min, dfd_max, dfd_avg = inf, -inf, 0.0

                for _ in range(TEST_ITERATIONS):
                    privatised_df = m.privatise_trajectory(gdf, eps, delta)
                    privatised_pts = [
                        (pt.x, pt.y) for pt in privatised_df.geometry.values
                    ]

                    dfd_error = frechet_dist(gdf_pts, privatised_pts)
                    dtw_error = dtw(gdf_pts, privatised_pts).normalizedDistance

                    dtw_min = min(dtw_min, dtw_error)
                    dtw_max = max(dtw_max, dtw_error)
                    dtw_avg += dtw_error

                    dfd_min = min(dfd_min, dfd_error)
                    dfd_max = max(dfd_max, dfd_error)
                    dfd_avg += dfd_error

                dtw_avg /= TEST_ITERATIONS
                dfd_avg /= TEST_ITERATIONS

                res["eps"].append(eps)
                res["dfd"].append((dfd_avg, dfd_min, dfd_max))
                res["dtw"].append((dtw_avg, dtw_min, dtw_max))

            # Plot the results for the given method
            dtw_ax.errorbar(
                res["eps"],
                [y[0] for y in res["dtw"]],
                yerr=(
                    np.subtract([y[0] for y in res["dtw"]], [y[1] for y in res["dtw"]]),
                    np.subtract([y[2] for y in res["dtw"]], [y[0] for y in res["dtw"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )
            dfd_ax.errorbar(
                res["eps"],
                [y[0] for y in res["dfd"]],
                yerr=(
                    np.subtract([y[0] for y in res["dfd"]], [y[1] for y in res["dfd"]]),
                    np.subtract([y[2] for y in res["dfd"]], [y[0] for y in res["dfd"]]),
                ),
                label=f"{m.NAME} Mean Error",
                linewidth=1,
                linestyle="-",
                color=m.COLOR,
                marker="*",
                markersize=8,
                markerfacecolor=m.COLOR,
                capsize=5,
                elinewidth=0.75,
            )

        # Configure the axes labels
        dtw_ax.set_title(f"Normalised Dynamic Time Warping Error", y=0.9)
        dtw_ax.set_xlabel("Epsilon (ε)")
        dtw_ax.set_ylabel("Error (m)")
        dtw_ax.set_xscale("log")
        dtw_ax.set_xticks(PRESET_EPSILONS)
        dtw_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        dfd_ax.set_title(f"Discrete Fréchet Distance Error", y=0.9)
        dfd_ax.set_xlabel("Epsilon (ε)")
        dfd_ax.set_ylabel("Error (m)")
        dfd_ax.set_xscale("log")
        dfd_ax.set_xticks(PRESET_EPSILONS)
        dfd_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))

        # Configure the figure and layout of the subplots
        fig.suptitle(
            f"Trajectory Error Calculations for Varying Epsilon (N: {TEST_ITERATIONS}, δ: {delta})"
        )
        fig.legend(
            handles=dtw_ax.get_legend_handles_labels()[0],
            labels=dtw_ax.get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.subplots_adjust(wspace=0.2, right=0.85)

        plt.show()

    def command_privatise_varying_epsilon(self):
        """
        Privatise a given ship's trajectory with varying epsilon
        """
        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        delta = float(input("Enter a value for delta (δ): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]
        pda = PointwiseDistanceAngle(self.florida_coastline)

        # Run the privitisation against the 3 current schemes available
        sdd_gdfs = [
            SampleDistanceDirection().privatise_trajectory(gdf, eps=epsilon, delta=0.0)
            for epsilon in PRESET_EPSILONS
        ]
        pda_gdfs = [
            pda.privatise_trajectory(gdf, eps=epsilon, delta=delta)
            for epsilon in PRESET_EPSILONS
        ]

        # Reduce the polygons we plot by forming a bbox of the min & max lat and long
        gdfs = [gdf, *sdd_gdfs, *pda_gdfs]
        # gdfs = [gdf, *pda_gdfs]
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

        # 2 rows of 3 columns with shared axes
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))

        for i in range(len(PRESET_EPSILONS)):
            ax: plt.Axes = axs[i // 3, i % 3]  # Current axis to plot on

            # Plot the trajectories
            ax.plot(
                gdf.geometry.x,
                gdf.geometry.y,
                linewidth=1,
                linestyle="--",
                color="green",
                marker="o",
                markersize=5,
                markerfacecolor="green",
                label="Original",
            )
            ax.plot(
                sdd_gdfs[i].geometry.x,
                sdd_gdfs[i].geometry.y,
                linewidth=1,
                linestyle="--",
                color="purple",
                marker="^",
                markersize=5,
                markerfacecolor="purple",
                label="SDD",
            )
            ax.plot(
                pda_gdfs[i].geometry.x,
                pda_gdfs[i].geometry.y,
                linewidth=1,
                linestyle="--",
                color="blue",
                marker="s",
                markersize=5,
                markerfacecolor="blue",
                label="PDA",
            )

            # Plot the coastline if applicable
            self.florida_coastline.cx[min_lon:max_lon, min_lat:max_lat].plot(
                ax=ax, color="black", alpha=0.5
            )

            # Configure the axes labels
            ax.set_title(f"ε={PRESET_EPSILONS[i]}", y=0.9)
            ax.set_xlabel("Longitude (EPSG:3857)")
            ax.set_ylabel("Latitude (EPSG:3857)")
            ax.set_facecolor("azure")

            # Annotate the start and end points of the trajectory
            ax.annotate("Start", (gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y))
            ax.annotate("End", (gdf.geometry.iloc[-1].x, gdf.geometry.iloc[-1].y))

            # Format the axes to be 4 sig figs with sci notation
            ax.xaxis.set_major_formatter(CustomScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(CustomScalarFormatter(useMathText=True))

        for ax in axs.flat:
            ax.label_outer()

        # Configure the figure and layout of the subplots
        fig.suptitle(f"Privatisation of Ship {ship_id} Against Varying ε (δ: {delta})")
        fig.legend(
            handles=axs[0, 0].get_legend_handles_labels()[0],
            labels=axs[0, 0].get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.axis("square")
        plt.xticks(plt.xticks()[0][0::2])
        plt.yticks(plt.yticks()[0][0::2])
        plt.ticklabel_format(useOffset=False)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0, bottom=0.1, right=0.85)
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

        Shortish trajectories:
            - 258288000
            - 368004120
            - 367555540
            - 367006650
        """

        # Filter the data to only include the given ship
        ship_id = int(input("Enter a ship ID (MMSI): "))
        epsilon = float(input("Enter a value for epsilon (ε): "))
        delta = float(input("Enter a value for delta (δ): "))

        gdf = self.gdf.loc[self.gdf["MMSI"] == ship_id]

        # Run the privitisation against the 3 current schemes available
        sdd_gdf = SampleDistanceDirection().privatise_trajectory(
            gdf, eps=epsilon, delta=0.0
        )
        pda_gdf = PointwiseDistanceAngle().privatise_trajectory(
            gdf, eps=epsilon, delta=delta
        )
        # avoid_gdf = AvoidantSamplingMethod(self.florida_coastline).privatise_trajectory(
        #     gdf, eps=epsilon
        # )
        # postprocessed_avoid_gdf = AvoidantSamplingMethod(
        #     self.florida_coastline
        # ).postprocess_result(avoid_gdf, 200)

        # Reduce the polygons we plot by forming a bbox of the min & max lat and long
        gdfs = [gdf, sdd_gdf, pda_gdf]
        min_lon, min_lat, max_lon, max_lat = (
            np.nanmin([df.geometry.x.min() for df in gdfs]),
            np.nanmin([df.geometry.y.min() for df in gdfs]),
            np.nanmax([df.geometry.x.max() for df in gdfs]),
            np.nanmax([df.geometry.y.max() for df in gdfs]),
        )

        # Clear the plot
        plt.cla()
        plt.clf()
        plt.close()

        # Set the axes labels and the title of the plot
        plt.xlabel("Lon (Mercator)")
        plt.ylabel("Lat (Mercator)")
        plt.title(f"Privatisation of Ship {ship_id} (ε: {epsilon}, δ: {delta})")

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
        plt.plot(
            sdd_gdf.geometry.x,
            sdd_gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="purple",
            marker="^",
            markersize=5,
            markerfacecolor="purple",
            label="SDD Trajectory",
        )
        plt.plot(
            pda_gdf.geometry.x,
            pda_gdf.geometry.y,
            linewidth=1,
            linestyle="--",
            color="blue",
            marker="s",
            markersize=5,
            markerfacecolor="blue",
            label="PDA Trajectory",
        )
        # plt.plot(
        #     avoid_gdf.geometry.x,
        #     avoid_gdf.geometry.y,
        #     linewidth=1,
        #     linestyle="--",
        #     color="orange",
        #     marker="o",
        #     markersize=5,
        #     markerfacecolor="orange",
        #     label="Avoidant Sampling Method",
        # )
        # plt.plot(
        #     postprocessed_avoid_gdf.geometry.x,
        #     postprocessed_avoid_gdf.geometry.y,
        #     linewidth=1,
        #     linestyle="--",
        #     color="indianred",
        #     marker="o",
        #     markersize=5,
        #     markerfacecolor="indianred",
        #     label="Postprocessed Avoidant Sampling Method",
        # )
        self.florida_coastline.cx[min_lon:max_lon, min_lat:max_lat].plot(
            ax=plt.gca(), color="black", alpha=0.5
        )

        # Annotate the start and end points of the trajectory
        plt.annotate("Start", (gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y))
        plt.annotate("End", (gdf.geometry.iloc[-1].x, gdf.geometry.iloc[-1].y))

        # Configure the display of the plot and show
        plt.axis("square")
        plt.ticklabel_format(useOffset=False)
        plt.gca().set_facecolor("azure")
        plt.legend()
        plt.show()

    def command_privatise_preset(self):
        """
        Display a plot of the given ship's trajectory for 4
        different pre-selected MMSIs

        - Short trajectory in open water (258288000)
        - Long trajectory in open water (316038559)
        - Short trajectory in a river (near landmasses) (367637340)
        - Long trajectory in a river (near landmasses) (368011140)
        """

        def on_pick(event):
            """
            Toggle the visibility of the line and corresponding
            entries in the legend of the plot
            """
            legend_line = event.artist
            original_lines, legend_text = legend_map[legend_line]
            visible = not original_lines[0].get_visible()

            for line in original_lines:
                line.set_visible(visible)

            legend_line.set_alpha(1.0 if visible else 0)
            legend_text.set_alpha(1.0 if visible else 0)

            fig.canvas.draw()

        # Get inputs for epsilon and delta
        epsilon = float(input("Enter a value for epsilon (ε): "))
        delta = float(input("Enter a value for delta (δ): "))

        # Create a 2x2 grid of plots
        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        lines = {
            "real": [],
            "sdd": [],
            "postprocessed_pda": [],
            "pda": [],
        }

        for i, mmsi in enumerate(PRESET_MMSIS):
            gdf = self.gdf.loc[self.gdf["MMSI"] == mmsi]  # Filter by current MMSI
            pda = PointwiseDistanceAngle(self.florida_coastline)

            # Run the privatisation against all schemes
            sdd_gdf = SampleDistanceDirection().privatise_trajectory(
                gdf, eps=epsilon, delta=0.0
            )
            pda_gdf = pda.privatise_trajectory(gdf, eps=epsilon, delta=delta)
            pda_gdf_postprocessing = pda.postprocess_result(pda_gdf, 200)

            # Reduce the polygons we plot by forming a bbox of the min & max lat and long
            gdfs = [gdf, sdd_gdf, pda_gdf]
            min_lon, min_lat, max_lon, max_lat = (
                np.min([df.geometry.x.min() for df in gdfs]),
                np.min([df.geometry.y.min() for df in gdfs]),
                np.max([df.geometry.x.max() for df in gdfs]),
                np.max([df.geometry.y.max() for df in gdfs]),
            )

            # Current axis to plot on
            ax: plt.Axes = axs[i // 2, i % 2]

            # Plot each of the trajectories
            lines["real"].append(
                ax.plot(
                    gdf.geometry.x,
                    gdf.geometry.y,
                    linewidth=1,
                    linestyle="--",
                    color="green",
                    marker="o",
                    markersize=5,
                    markerfacecolor="green",
                    label="Original",
                )[0]
            )
            lines["sdd"].append(
                ax.plot(
                    sdd_gdf.geometry.x,
                    sdd_gdf.geometry.y,
                    linewidth=1,
                    linestyle="--",
                    color="purple",
                    marker="^",
                    markersize=5,
                    markerfacecolor="purple",
                    label="SDD",
                )[0]
            )
            lines["postprocessed_pda"].append(
                ax.plot(
                    pda_gdf_postprocessing.geometry.x,
                    pda_gdf_postprocessing.geometry.y,
                    linewidth=1,
                    linestyle="--",
                    color="indianred",
                    marker="P",
                    markersize=5,
                    markerfacecolor="indianred",
                    label="Postprocessed",
                )[0]
            )
            lines["pda"].append(
                ax.plot(
                    pda_gdf.geometry.x,
                    pda_gdf.geometry.y,
                    linewidth=1,
                    linestyle="--",
                    color="blue",
                    marker="s",
                    markersize=5,
                    markerfacecolor="blue",
                    label="PDA",
                )[0]
            )

            # Plot the coastline if applicable
            self.florida_coastline.cx[min_lon:max_lon, min_lat:max_lat].plot(
                ax=ax, color="black", alpha=0.5
            )

            # Configure the axes labels
            ax.set_title(f"Ship {mmsi}", y=0.9)
            ax.set_xlabel("Longitude (EPSG:3857)")
            ax.set_ylabel("Latitude (EPSG:3857)")
            ax.set_facecolor("azure")
            ax.axis("square")

            # Format the axes to be 4 sig figs with sci notation
            ax.xaxis.set_major_formatter(CustomScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(CustomScalarFormatter(useMathText=True))

            # Annotate the start and end points of the trajectory
            ax.annotate("Start", (gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y))
            ax.annotate("End", (gdf.geometry.iloc[-1].x, gdf.geometry.iloc[-1].y))

        # Configure the figure and layout of the subplots
        fig.suptitle(f"Privatisation of Ship Trajectories (ε: {epsilon}, δ: {delta})")
        legend = fig.legend(
            handles=axs[0, 0].get_legend_handles_labels()[0],
            labels=axs[0, 0].get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.xticks(plt.xticks()[0][0::2])
        plt.yticks(plt.yticks()[0][0::2])
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, right=0.85)

        # Make each entry of the legend clickable, toggling the display of the corresponding plot
        legend_map = {}
        for legend_line, legend_text, original_lines in zip(
            legend.get_lines(), legend.get_texts(), lines.values()
        ):
            legend_line.set_picker(5)
            legend_map[legend_line] = (original_lines, legend_text)

        plt.connect("pick_event", on_pick)
        plt.show()

    def command_error_preset(self):
        """
        Display a plot of the given ship's DTW & DFD
        error for each of the 4 different pre-selected MMSIs

        - Short trajectory in open water (258288000)
        - Long trajectory in open water (316038559)
        - Short trajectory in a river (near landmasses) (367637340)
        - Long trajectory in a river (near landmasses) (368011140)
        """

        # Get inputs for epsilon and delta
        epsilon = float(input("Enter a value for epsilon (ε): "))
        delta = float(input("Enter a value for delta (δ): "))

        # Clear and reset the plot
        plt.cla()
        plt.clf()
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(14, 10))
        dtw_ax: plt.Axes = axs[0]
        dfd_ax: plt.Axes = axs[1]

        data: Dict[str, List[float]] = {
            "x": np.arange(len(PRESET_MMSIS)),
            "sdd_dtw": [],
            "sdd_dfd": [],
            "pda_dtw": [],
            "pda_dfd": [],
        }

        for mmsi in PRESET_MMSIS:
            # Filter the data to only include the given ship
            gdf = self.gdf.loc[self.gdf["MMSI"] == mmsi]
            gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

            # Use both privatisation schemes
            pda = PointwiseDistanceAngle(self.florida_coastline)
            sdd = SampleDistanceDirection()

            # Store the min, max and average error for each scheme
            sdd_dtw_min, sdd_dtw_max, sdd_dtw_avg = inf, -inf, 0.0
            sdd_dfd_min, sdd_dfd_max, sdd_dfd_avg = inf, -inf, 0.0
            pda_dtw_min, pda_dtw_max, pda_dtw_avg = inf, -inf, 0.0
            pda_dfd_min, pda_dfd_max, pda_dfd_avg = inf, -inf, 0.0

            # total_dist = np.sum(
            #     [
            #         pt.distance(gdf.geometry.iloc[i + 1])
            #         for i, pt in enumerate(gdf.geometry.values[:-1])
            #     ]
            # )

            for _ in range(TEST_ITERATIONS):
                # Run the privatisation against both schemes
                sdd_gdf = sdd.privatise_trajectory(gdf, eps=epsilon, delta=0.0)
                pda_gdf = pda.privatise_trajectory(gdf, eps=epsilon, delta=delta)

                sdd_pts = [(pt.x, pt.y) for pt in sdd_gdf.geometry.values]
                pda_pts = [(pt.x, pt.y) for pt in pda_gdf.geometry.values]

                # Calculate the normalised error using DTW and error using DFD
                sdd_dtw_error = dtw(sdd_pts, gdf_pts).normalizedDistance
                pda_dtw_error = dtw(pda_pts, gdf_pts).normalizedDistance
                sdd_dfd_error = frechet_dist(gdf_pts, sdd_pts)
                pda_dfd_error = frechet_dist(gdf_pts, pda_pts)

                # Update the min, max and average error for DTW
                sdd_dtw_min = min(sdd_dtw_min, sdd_dtw_error)
                sdd_dtw_max = max(sdd_dtw_max, sdd_dtw_error)
                sdd_dtw_avg += sdd_dtw_error

                pda_dtw_min = min(pda_dtw_min, pda_dtw_error)
                pda_dtw_max = max(pda_dtw_max, pda_dtw_error)
                pda_dtw_avg += pda_dtw_error

                # Update the min, max and average error for DFD
                sdd_dfd_min = min(sdd_dfd_min, sdd_dfd_error)
                sdd_dfd_max = max(sdd_dfd_max, sdd_dfd_error)
                sdd_dfd_avg += sdd_dfd_error

                pda_dfd_min = min(pda_dfd_min, pda_dfd_error)
                pda_dfd_max = max(pda_dfd_max, pda_dfd_error)
                pda_dfd_avg += pda_dfd_error

            # Calculate the average error for each scheme
            sdd_dtw_avg /= TEST_ITERATIONS
            sdd_dfd_avg /= TEST_ITERATIONS
            pda_dtw_avg /= TEST_ITERATIONS
            pda_dfd_avg /= TEST_ITERATIONS

            data["sdd_dtw"].append((sdd_dtw_avg, sdd_dtw_min, sdd_dtw_max))
            data["sdd_dfd"].append((sdd_dfd_avg, sdd_dfd_min, sdd_dfd_max))
            data["pda_dtw"].append((pda_dtw_avg, pda_dtw_min, pda_dtw_max))
            data["pda_dfd"].append((pda_dfd_avg, pda_dfd_min, pda_dfd_max))

        # Plot a bar chart with the average, min and max error for each scheme
        dtw_ax.bar(
            [x - 0.2 for x in data["x"]],
            [y[0] for y in data["pda_dtw"]],
            yerr=(
                np.subtract(
                    [y[0] for y in data["pda_dtw"]], [y[1] for y in data["pda_dtw"]]
                ),
                np.subtract(
                    [y[2] for y in data["pda_dtw"]], [y[1] for y in data["pda_dtw"]]
                ),
            ),
            width=0.4,
            color="blue",
            label="PDA Mean Error",
            linewidth=1,
            linestyle="-",
            capsize=5,
            error_kw={"elinewidth": 2, "capthick": 2},
        )
        dtw_ax.bar(
            [x + 0.2 for x in data["x"]],
            [y[0] for y in data["sdd_dtw"]],
            yerr=(
                np.subtract(
                    [y[0] for y in data["sdd_dtw"]], [y[1] for y in data["sdd_dtw"]]
                ),
                np.subtract(
                    [y[2] for y in data["sdd_dtw"]], [y[1] for y in data["sdd_dtw"]]
                ),
            ),
            width=0.4,
            color="purple",
            label="SDD Mean Error",
            linewidth=1,
            linestyle="-",
            capsize=8,
            error_kw={"elinewidth": 2, "capthick": 2},
        )
        dfd_ax.bar(
            [x - 0.2 for x in data["x"]],
            [y[0] for y in data["pda_dfd"]],
            yerr=(
                np.subtract(
                    [y[0] for y in data["pda_dfd"]], [y[1] for y in data["pda_dfd"]]
                ),
                np.subtract(
                    [y[2] for y in data["pda_dfd"]], [y[1] for y in data["pda_dfd"]]
                ),
            ),
            width=0.4,
            color="blue",
            label="PDA Mean Error",
            linewidth=1,
            linestyle="-",
            capsize=8,
            error_kw={"elinewidth": 2, "capthick": 2},
        )
        dfd_ax.bar(
            [x + 0.2 for x in data["x"]],
            [y[0] for y in data["sdd_dfd"]],
            yerr=(
                np.subtract(
                    [y[0] for y in data["sdd_dfd"]], [y[1] for y in data["sdd_dfd"]]
                ),
                np.subtract(
                    [y[2] for y in data["sdd_dfd"]], [y[1] for y in data["sdd_dfd"]]
                ),
            ),
            width=0.4,
            color="purple",
            label="SDD Mean Error",
            linewidth=1,
            linestyle="-",
            capsize=8,
            error_kw={"elinewidth": 2, "capthick": 2},
        )

        # Configure the axes labels
        dtw_ax.set_title(f"Normalised Dynamic Time Warping Error", y=0.9)
        dtw_ax.set_xlabel("Ship Number (MMSI)")
        dtw_ax.set_ylabel("Error (m)")
        dtw_ax.set_xticks(data["x"], PRESET_MMSIS)

        dfd_ax.set_title(f"Discrete Fréchet Distance Error", y=0.9)
        dfd_ax.set_xlabel("Ship Number (MMSI)")
        dfd_ax.set_ylabel("Error (m)")
        dfd_ax.set_xticks(data["x"], PRESET_MMSIS)

        # Configure the figure and layout of the subplots
        fig.suptitle(
            f"Trajectory Error Calculations (N: {TEST_ITERATIONS}, ε: {epsilon}, δ: {delta})"
        )
        fig.legend(
            handles=dtw_ax.get_legend_handles_labels()[0],
            labels=dtw_ax.get_legend_handles_labels()[1],
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
        )
        plt.subplots_adjust(wspace=0.2, right=0.85)

        print(data)
        plt.show()


if __name__ == "__main__":
    privatiser = Privatiser()

    while True:
        privatiser.run_command()
