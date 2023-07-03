import pandas as pd
import os
import numpy as np
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

DATASET_FOLDER = os.path.join(Path(__file__).parent.parent, "AIS_2019_01_01.csv")

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude


def sample_angle(eps: float, theta: float) -> float:
    """
    Samples an angle theta according to the pdf given by
    Pr(θ) = C*exp(ε*(2π - |α_i-θ|)/4π)
    Here we have:
    Δu = 2π, u(α, θ) = 2π - |α-θ| where θ, α ∈ [0, 2π]
    1/C = 4π[2e^(ε/2)-e^(ε(2π-θ)/4π)-e^(εθ/4π)]/ε
    which gives a pdf with area 1.

    The utility function generates maximum utility when the angle
    between the ship's projected angle to the next point and its
    noisy estimate is 0.
    """
    exp_factor = 4 * np.pi / eps

    def inverse_cdf(X: float, eps: float, theta: float) -> float:
        # Constant from PDF
        C = (
            2 * np.exp(eps / 2)
            - np.exp((2 * np.pi - theta) / exp_factor)
            - np.exp(theta / exp_factor)
        )

        # This is the inverted piece for when angles are in the range [0, theta]
        piecewise_A = (
            theta
            - 2 * np.pi
            + np.log(C * X + np.exp((2 * np.pi - theta) / exp_factor)) * exp_factor
        )

        # This is the inverted piece for when angles are in the range [theta, 2pi]
        piecewise_B = (
            theta
            + 2 * np.pi
            - np.log(
                -C * X + 2 * np.exp(eps / 2) - np.exp((2 * np.pi - theta) / exp_factor)
            )
            * exp_factor
        )

        # Return the correct piecewise function depending on the resultant value of the inverse
        if 0 <= piecewise_A < theta:
            return piecewise_A
        elif theta <= piecewise_B <= 2 * np.pi:
            return piecewise_B

    # Utilise the inverse transform sampling method to sample from the distribution
    X = np.random.uniform(0, 1)

    # Use the inverse CDF to sample from the distribution
    return inverse_cdf(X, eps, theta)


def sample_distance(eps: float, r: float) -> float:
    """
    Samples a distance according to the pdf given by
    Pr(r) = C*exp(ε*(r-x)/2r)
    Here we have:
    Δu = r, u(r, x) = r - x where x ∈ [0, r]
    1/C = 2r(e^(ε/2)-1)/ε
    which gives a pdf with area 1.

    The utility function generates maximum utility when the distance
    between ship's current private position and the next point in the location
    data is as minimal as possible
    """

    def inverse_cdf(X: float, eps: float, r: float) -> float:
        return r - 2 * r * np.log(np.exp(eps / 2) + (1 - np.exp(eps / 2)) * X) / eps

    # Utilise the inverse transform sampling method to sample from the distribution
    X = np.random.uniform(0, 1)
    x = np.linspace(0, 1, 100)
    y = [inverse_cdf(x, eps, r) for x in x]
    plt.plot(x, y)
    plt.show()

    # Use the inverse CDF to sample from the distribution
    return inverse_cdf(X, eps, r)

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
