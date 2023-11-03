from typing import Dict
import numpy as np
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.lines import Line2D  # type: ignore

from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection

from common import PRESET_MMSIS, CustomScalarFormatter


def command_privatise_preset(
    initial_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame
):
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

    lines: Dict[str, Line2D] = {
        "real": [],
        "sdd": [],
        "postprocessed_pda": [],
        "pda": [],
    }

    for i, mmsi in enumerate(PRESET_MMSIS):
        # Filter the data to only include the given ship
        gdf = initial_gdf.loc[initial_gdf["MMSI"] == mmsi]
        pda = PointwiseDistanceAngle(boundaries=boundaries)

        # Run the privatisation against all schemes
        sdd_gdf = SampleDistanceDirection().privatise_trajectory(
            gdf, eps=epsilon, delta=delta
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
        boundaries.cx[min_lon:max_lon, min_lat:max_lat].plot(
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
