import numpy as np
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection

from common import PRESET_EPSILONS, CustomScalarFormatter


def command_privatise_varying_epsilon(
    initial_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame
):
    """
    Conducts an experiment for a given input MMSI and delta value.
    Varies the epsilon value and privatises the route for each method
    for each value of epsilon.

    Results are plotted on a graph in a 2 x 3 grid.
    """

    # Filter the data to only include the given ship
    ship_id = int(input("Enter a ship ID (MMSI): "))
    delta = float(input("Enter a value for delta (δ): "))

    # Filter the data to only include the given ship
    gdf = initial_gdf.loc[initial_gdf["MMSI"] == ship_id]
    pda = PointwiseDistanceAngle(boundaries=boundaries)

    # Run the privitisation against the 3 current schemes available
    sdd_gdfs = [
        SampleDistanceDirection().privatise_trajectory(gdf, eps=epsilon, delta=delta)
        for epsilon in PRESET_EPSILONS
    ]
    pda_gdfs = [
        pda.privatise_trajectory(gdf, eps=epsilon, delta=delta)
        for epsilon in PRESET_EPSILONS
    ]

    # Reduce the polygons we plot by forming a bbox of the min & max lat and long
    gdfs = [gdf, *sdd_gdfs, *pda_gdfs]
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
        boundaries.cx[min_lon:max_lon, min_lat:max_lat].plot(
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

    # Label the outsides of the subplots
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
