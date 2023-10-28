import geopandas as gpd  # type: ignore
from shapely.geometry import Point, LineString  # type: ignore
import numpy as np
from math import inf
from shapely.ops import nearest_points  # type: ignore
from shapely import offset_curve  # type: ignore

from typing import Dict, List
from sample_method import SampleMethod
from constants import WEB_MERCATOR


class PointwiseDistanceAngle(SampleMethod):
    """
    Contains the Pointwise Distance Angle sampling functions.
    """

    COLOR = "blue"
    NAME = "PDA"
    boundaries: gpd.GeoDataFrame  # This is the loaded shape of the coastline

    def __init__(self, boundaries: gpd.GeoDataFrame):
        self.boundaries = boundaries

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

    def _raycast_intersection(self, linestring: LineString) -> bool:
        """
        Checks if the given linestring intersects with the coastline.

        Returns True if the linestring intersects an even number of times with the coastline.
        """

        line_gdf = gpd.GeoDataFrame(geometry=[linestring], crs=WEB_MERCATOR)
        intersection = line_gdf.overlay(
            self.boundaries, how="intersection", keep_geom_type=False
        )

        # You get 0 or more geometries that are either a Point or MultiPoint
        intersection_pts = 0
        for geometry in intersection.geometry:
            if geometry.geom_type == "Point":
                intersection_pts += 1
            else:
                intersection_pts += len(geometry.geoms)

        return intersection_pts % 2 == 0

    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float, delta: float
    ) -> gpd.GeoDataFrame:
        """
        Privatises the trajectory of a ship using our developed method.
        """

        # Preprocess the trajectory
        gdf = self._preprocess_trajectory(gdf, delta)

        # Privatised route starts and ends at the same location as the real route
        privatised: Dict[str, List[Point | bool]] = {
            "geometry": [],
            "valid": [],
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
            new_point = Point(
                current_point.x + sampled_distance * np.cos(sampled_angle),
                current_point.y + sampled_distance * np.sin(sampled_angle),
            )
            valid = self._raycast_intersection(LineString([current_point, new_point]))
            privatised["geometry"].append(new_point)
            privatised["valid"].append(valid)

        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)

    def postprocess_result(
        self, trajectory: gpd.GeoDataFrame, buffer: float = 0
    ) -> gpd.GeoDataFrame:
        trajectory = trajectory.copy()

        for i in range(len(trajectory)):
            if not trajectory.valid.iloc[i]:
                # If the point is invalid, find the closest valid point
                # on the boundary from the privatised point and replace it
                pt = trajectory.geometry.iloc[i]
                closest_pt: Point = None
                closest_distance = inf

                # For each line in the boundary, find the closest point
                for line in self.boundaries.geometry:
                    try:
                        nearest_pt = nearest_points(line, pt)[0]
                        dist = nearest_pt.distance(pt)

                        if dist < closest_distance:
                            closest_pt = nearest_pt
                            closest_distance = dist
                    except ValueError:
                        # Skip empty geometries
                        pass

                # Calculate the replacement point by adding buffer distance to the vector from the original point to the closest point
                print(f"Replacing {pt} with {closest_pt} + {buffer}")
                v = np.array(
                    [
                        closest_pt.x - pt.x,
                        closest_pt.y - pt.y,
                    ]
                )
                new_pt = Point(
                    closest_pt.x + buffer * v[0] / np.linalg.norm(v),
                    closest_pt.y + buffer * v[1] / np.linalg.norm(v),
                )
                trajectory.geometry.iloc[i] = new_pt

        return trajectory.drop(columns=["valid"])
