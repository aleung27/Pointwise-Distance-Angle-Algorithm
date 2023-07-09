from sample_method import SampleMethod

import numpy as np
import geopandas as gpd  # type: ignore
from shapely.geometry import Point  # type: ignore

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps


class AdapatedSamplingMethod(SampleMethod):
    """
    Contains the Adapted Sampling Method sampling functions.
    """

    def _sample_angle(self, eps: float, theta: float) -> float:
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
                    -C * X
                    + 2 * np.exp(eps / 2)
                    - np.exp((2 * np.pi - theta) / exp_factor)
                )
                * exp_factor
            )

            # Return the correct piecewise function depending on the resultant value of the inverse
            if 0 <= piecewise_A < theta:
                return piecewise_A
            elif theta <= piecewise_B <= 2 * np.pi:
                return piecewise_B
            else:
                raise ValueError(
                    f"Invalid value for piecewise function: {piecewise_A} or {piecewise_B}"
                )

        # Utilise the inverse transform sampling method to sample from the distribution
        X = np.random.uniform(0, 1)

        # Use the inverse CDF to sample from the distribution
        return inverse_cdf(X, eps, theta)

    def _sample_distance(self, eps: float, r: float) -> float:
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

        # Use the inverse CDF to sample from the distribution
        return inverse_cdf(X, eps, r)

    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float
    ) -> gpd.GeoDataFrame:
        """
        Privatises the trajectory of a ship using the Adapted Sampling Method.
        """

        # Privatised route starts and ends at the same location as the real route
        privatised = {
            "geometry": [gdf.geometry.iloc[0]],
        }

        for i in range(1, len(gdf) - 1):
            # The current (real) point and the last (privatised) point
            current_point: Point = gdf.geometry.iloc[i]
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
            sampled_distance = self._sample_distance(eps, distance)
            sampled_angle = self._sample_angle(eps, angle)
            # print(
            #     f"Distance {distance} & angle {angle} -> Distance {sampled_distance} & angle {sampled_angle}"
            # )

            # Calculate the new point at angle sampled_angle in a circle of radius sampled_distance
            privatised["geometry"].append(
                Point(
                    current_point.x + sampled_distance * np.cos(sampled_angle),
                    current_point.y + sampled_distance * np.sin(sampled_angle),
                )
            )

        privatised["geometry"].append(gdf.geometry.iloc[-1])
        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)
