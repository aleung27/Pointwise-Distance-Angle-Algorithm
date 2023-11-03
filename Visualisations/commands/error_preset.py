import numpy as np
from typing import List, Any, TypedDict, Tuple

import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection

from dtw import dtw  # type: ignore
from similaritymeasures import frechet_dist  # type: ignore
from math import inf

from common import PRESET_MMSIS, TEST_ITERATIONS


class DataDict(TypedDict):
    x: np.ndarray[Any, Any]
    sdd_dtw: List[Tuple[float, float, float]]
    sdd_dfd: List[Tuple[float, float, float]]
    pda_dtw: List[Tuple[float, float, float]]
    pda_dfd: List[Tuple[float, float, float]]


def command_error_preset(initial_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame):
    """
    Display a plot of the given ship's DTW & DFD
    error for each of the 4 different pre-selected MMSIs
    and a given epsilon and delta value.

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

    # Store the data for each ship
    data: DataDict = {
        "x": np.arange(len(PRESET_MMSIS)),
        "sdd_dtw": [],
        "sdd_dfd": [],
        "pda_dtw": [],
        "pda_dfd": [],
    }

    for mmsi in PRESET_MMSIS:
        # Filter the data to only include the given ship
        gdf = initial_gdf.loc[initial_gdf["MMSI"] == mmsi]
        gdf_pts = [(pt.x, pt.y) for pt in gdf.geometry.values]

        # Use both privatisation schemes
        pda = PointwiseDistanceAngle(boundaries=boundaries)
        sdd = SampleDistanceDirection()

        # Store the min, max and average error for each scheme
        sdd_dtw_min, sdd_dtw_max, sdd_dtw_avg = inf, -inf, 0.0
        sdd_dfd_min, sdd_dfd_max, sdd_dfd_avg = inf, -inf, 0.0
        pda_dtw_min, pda_dtw_max, pda_dtw_avg = inf, -inf, 0.0
        pda_dfd_min, pda_dfd_max, pda_dfd_avg = inf, -inf, 0.0

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
