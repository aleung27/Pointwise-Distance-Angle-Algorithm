import numpy as np
import geopandas as gpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from pointwise_distance_angle import PointwiseDistanceAngle
from sdd import SampleDistanceDirection


def command_privatise(initial_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame):
    """
    Privatise a given ship's trajectory using the available different methods
    and plot the results on a graph. Some examples of trajectories include

    Florida (matanzas river) Trajectories:
        - 367597490
        - 367181030
        - 368057420

    Small open ocean trajectory:
        - 368048550
        - 258288000
        - 368004120
        - 367555540
        - 367006650
    """

    # Get the inputs for the ship ID, epsilon and delta
    ship_id = int(input("Enter a ship ID (MMSI): "))
    epsilon = float(input("Enter a value for epsilon (ε): "))
    delta = float(input("Enter a value for delta (δ): "))

    # Filter the data to only include the given ship
    gdf = initial_gdf.loc[initial_gdf["MMSI"] == ship_id]
    pda = PointwiseDistanceAngle(boundaries=boundaries)

    # Run the privitisation against the 3 current schemes available
    sdd_gdf = SampleDistanceDirection().privatise_trajectory(
        gdf, eps=epsilon, delta=delta
    )
    pda_gdf = pda.privatise_trajectory(gdf, eps=epsilon, delta=delta)
    postprocessed_pda_gdf = pda.postprocess_result(pda_gdf, 200)

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
    plt.plot(
        postprocessed_pda_gdf.geometry.x,
        postprocessed_pda_gdf.geometry.y,
        linewidth=1,
        linestyle="--",
        color="indianred",
        marker="o",
        markersize=5,
        markerfacecolor="indianred",
        label="Postprocessed PDA Trajectory",
    )

    boundaries.cx[min_lon:max_lon, min_lat:max_lat].plot(
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
