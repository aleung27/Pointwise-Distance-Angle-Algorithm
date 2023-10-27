import geopandas as gpd  # type: ignore
from shapely.geometry import Point  # type: ignore
import numpy as np
from typing import Dict
from sample_method import SampleMethod
from constants import WEB_MERCATOR


class PointwiseDistanceAngle(SampleMethod):
    """
    Contains the Pointwise Distance Angle sampling functions.
    """

    COLOR = "blue"
    NAME = "PDA"

    def _sample_angle(self) -> float:
        """
        Return an angle between [0, 2π] uniformly
        """
        return np.random.uniform(0, 2 * np.pi)

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

    def _preprocess_trajectory(
        self, gdf: gpd.GeoDataFrame, delta: float
    ) -> gpd.GeoDataFrame:
        """
        Preprocess the trajectory, removing points which are closer than 2/δ to each other.
        """

        # The first point is always included
        preprocessed = {
            "geometry": [gdf.geometry.iloc[0]],
        }

        for i in range(1, len(gdf)):
            # The current (real) point and the last (privatised) point
            current_point: Point = gdf.geometry.iloc[i]
            last_point: Point = preprocessed["geometry"][-1]

            # Get the distance and angle from the vector between the two points
            v = np.array(
                [
                    current_point.x - last_point.x,
                    current_point.y - last_point.y,
                ]
            )
            distance = float(np.linalg.norm(v))

            # If the distance is greater than 2/δ, add the point to the preprocessed trajectory
            if distance >= 2 / delta:
                preprocessed["geometry"].append(current_point)

        return gpd.GeoDataFrame(preprocessed, geometry="geometry", crs=WEB_MERCATOR)

    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float, delta: float
    ) -> gpd.GeoDataFrame:
        """
        Privatises the trajectory of a ship using our developed method.
        """

        # Preprocess the trajectory
        gdf = self._preprocess_trajectory(gdf, delta)

        # Privatised route starts and ends at the same location as the real route
        privatised: Dict[str, Point] = {
            "geometry": [],
        }

        for i in range(1, len(gdf)):
            # The current (real) point and the last (privatised) point
            current_point: Point = gdf.geometry.iloc[i]
            last_estimate: Point = (
                privatised["geometry"][-1]
                if privatised["geometry"]
                else gdf.geometry.iloc[0]
            )

            # Get the distance and angle from the vector between the two points
            v = np.array(
                [
                    current_point.x - last_estimate.x,
                    current_point.y - last_estimate.y,
                ]
            )
            distance = float(np.linalg.norm(v))

            # Sample a distance and angle from the given distributions
            sampled_distance = self._sample_distance(eps, distance)
            sampled_angle = self._sample_angle()
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

        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)
