import numpy as np
from typing import Dict, List, Any

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.ticker import StrMethodFormatter  # type: ignore

from sample_method import SampleMethod
from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection

from dtw import dtw  # type: ignore
from similaritymeasures import frechet_dist  # type: ignore
from math import inf

from common import PRESET_DELTAS, TEST_ITERATIONS


def command_error_varying_delta(
    initial_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame
):
    """
    Conducts an experiment for a given input MMSI and epsilon value.
    Varies the delta value and calculates the error for each method
    for each value of delta.

    Results are plotted on a graph.
    """

    # Methods to test - remove or add more here to see them in the plot
    methods: List[SampleMethod] = [
        SampleDistanceDirection(),
        PointwiseDistanceAngle(boundaries=boundaries),
    ]

    # Filter the data to only include the given ship
    ship_id = int(input("Enter a ship ID (MMSI): "))
    epsilon = float(input("Enter a value for epsilon (ε): "))

    # Filter the data to only include the given ship
    gdf = initial_gdf.loc[initial_gdf["MMSI"] == ship_id]
    gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

    # Clear the plot
    plt.cla()
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    dtw_ax: plt.Axes = axs[0]
    dfd_ax: plt.Axes = axs[1]

    # For each method, run through each test value of delta TEST_ITERATIONS times
    for m in methods:
        res: Dict[str, List[Any]] = {"delta": [], "dtw": [], "dfd": []}

        for delta in PRESET_DELTAS:
            # Store the min, max and avg for each error metric
            dtw_min, dtw_max, dtw_avg = inf, -inf, 0.0
            dfd_min, dfd_max, dfd_avg = inf, -inf, 0.0

            for _ in range(TEST_ITERATIONS):
                # Privatise the trajectory and calculate the error
                privatised_df = m.privatise_trajectory(gdf, epsilon, delta)
                privatised_pts = [(pt.x, pt.y) for pt in privatised_df.geometry.values]

                dfd_error = frechet_dist(privatised_pts, gdf_pts)
                dtw_error = dtw(privatised_pts, gdf_pts).normalizedDistance

                # Update the min, max and avg for each error metric
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

        # Plot the results for the given method as an errorbar plot
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
