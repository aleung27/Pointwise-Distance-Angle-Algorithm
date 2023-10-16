from sample_method import SampleMethod
from constants import WEB_MERCATOR
from math import inf
from typing import List, Tuple, Dict
import numpy as np
from scipy.stats import rv_continuous  # type: ignore
from shapely.geometry import Point, LineString  # type: ignore
from shapely.ops import nearest_points  # type: ignore
from shapely import offset_curve  # type: ignore
import geopandas as gpd  # type: ignore

RegionList = List[Tuple[float, float]]


class AvoidantSamplingMethod(SampleMethod):
    """
    Contains the Avoidant Sampling Method sampling functions. This method allows sampling
    an angle whilst specifying multiple disjoint regions to avoid.
    """

    COLOR = "orange"

    boundaries: gpd.GeoDataFrame  # This is the loaded shape of the coastline

    def __init__(self, boundaries: gpd.GeoDataFrame) -> None:
        self.boundaries = boundaries

    def _sample_angle(
        self,
        eps: float,
        theta: float,
        valid_regions: RegionList,
        invalid_regions: RegionList,
    ) -> float:
        """
        Samples an angle x using the utility function given that
        attempts to probabilistically avoid certain specified regions.
        We denote the set A as the set of pairwise angle ranges that
        we wish to avoid. These form the "avoidance" regions.

        The utility function is then given by
        u(x) = 1, x ∈ [0, 2π]\A (Denoted as B)
        u(x) = 0, x ∈ A
        Δu = 1

        Then we have the following value for 1/C
        1/C = ∑_{i=0}^|B| e^(ε/2)[B_i^U - B_i^L] + ∑_{i=0}^I(A) [A_i^U - A_i^L]

        The chance of avoidance approaches 100% as the value of ε increases.
        """

        class DistanceDistribution(rv_continuous):
            def __init__(
                self,
                eps: float,
                valid_regions: RegionList,
                invalid_regions: RegionList,
                *args,
                **kwargs,
            ) -> None:
                super().__init__(*args, **kwargs)
                self.eps = eps
                self.valid_regions = valid_regions
                self.invalid_regions = invalid_regions

            def _cdf(self, x, *args):
                complete_valid_region_sum = np.sum(
                    [r[1] - r[0] for r in self.valid_regions]
                ) * np.exp(self.eps / 2)
                complete_invalid_region_sum = np.sum(
                    [r[1] - r[0] for r in self.invalid_regions]
                )

                valid_region_sum = 0
                invalid_region_sum = 0
                for r in self.valid_regions:
                    if r[1] <= x:
                        # Region is completely below x
                        valid_region_sum += r[1] - r[0]
                    elif r[0] <= x <= r[1]:
                        # Region intersects x
                        valid_region_sum += x - r[0]

                for r in self.invalid_regions:
                    if r[1] <= x:
                        # Region is completely below x
                        invalid_region_sum += r[1] - r[0]
                    elif r[0] <= x <= r[1]:
                        # Region intersects x
                        invalid_region_sum += x - r[0]

                valid_region_sum *= np.exp(self.eps / 2)

                return (valid_region_sum + invalid_region_sum) / (
                    complete_valid_region_sum + complete_invalid_region_sum
                )

        distribution = DistanceDistribution(
            eps=eps,
            valid_regions=valid_regions,
            invalid_regions=invalid_regions,
            a=0,
            b=2 * np.pi,
        )

        return distribution.rvs()

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
        Privatises the trajectory of a ship using our developed method.
        """

        # Privatised route starts and ends at the same location as the real route
        privatised: Dict[str, List[Point | bool]] = {
            "geometry": [gdf.geometry.iloc[0]],
            "valid": [True],
        }

        for i in range(1, len(gdf) - 1):
            # The current (real) point and the last (privatised) point
            current_point: Point = gdf.geometry.iloc[i]
            last_estimate: Point = privatised["geometry"][-1]

            # Get the distance and angle from the vector between the two points
            # Angle is positive going counterclockwise from the positive x-axis
            v = np.array(
                [
                    current_point.x - last_estimate.x,
                    current_point.y - last_estimate.y,
                ]
            )
            distance = np.linalg.norm(v)
            angle = (np.arctan2(v[1], v[0]) + 2 * np.pi) % (2 * np.pi)

            # Sample a distance and angle from the given distributions
            sampled_distance = self._sample_distance(eps, distance)  # type: ignore
            valid_regions, invalid_regions = self.find_restricted_regions(
                current_point, sampled_distance
            )
            # print(valid_regions, invalid_regions)
            sampled_angle = self._sample_angle(
                eps, angle, valid_regions, invalid_regions
            )

            valid = True
            if any([ir[0] <= sampled_angle <= ir[1] for ir in invalid_regions]):
                # If the sampled angle lies in an invalid region, mark it for post processing
                valid = False

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
            privatised["valid"].append(valid)

        privatised["geometry"].append(gdf.geometry.iloc[-1])
        privatised["valid"].append(True)
        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)

    def _raycast_intersections(
        self,
        region: Tuple[float, float],
        intersecting_boundaries: gpd.GeoDataFrame,
        point: Point,
        distance: float,
    ) -> int:
        """
        Perform a raycast from from the given point to a point in the middle of the region lying on the circle's boundary.
        """

        # Get a point in the middle of the region that lies on the circle arc
        midpt = Point(
            point.x + distance * np.cos(np.mean(region)),
            point.y + distance * np.sin(np.mean(region)),
        )

        # Form a line string object from the centre of the circle to the midpoint
        line_gdf = gpd.GeoDataFrame(
            geometry=[LineString([point, midpt])], crs=WEB_MERCATOR
        )

        # Perform an intersection with the coastline to see how many times it crosses a segment
        intersection = line_gdf.overlay(
            intersecting_boundaries, how="intersection", keep_geom_type=False
        )

        # You get 0 or more geometries that are either a Point or MultiPoint
        intersection_pts = 0
        for geometry in intersection.geometry:
            if geometry.geom_type == "Point":
                intersection_pts += 1
            else:
                intersection_pts += len(geometry.geoms)

        return intersection_pts

    def find_restricted_regions(
        self, point: Point, distance: float
    ) -> Tuple[RegionList, RegionList]:
        """
        Given a point and a distance, find the regions of the circle that are restricted by the coastline.

        Assumptions:
        - Original point lies within the valid "region" <- can perform raycasting algorithm to check this?
        """

        # Polygon representing the point with radius given by distance
        buffered_point = point.buffer(distance)

        # Perform 2 intersections with the coastline map, yielding a MultiLineString of
        # coastline lying within the buffered point and a MultiPoint of the circle circumference intersection
        intersection_boundaries_gdf = gpd.overlay(
            self.boundaries,
            gpd.GeoDataFrame(geometry=[buffered_point], crs=WEB_MERCATOR),
            how="intersection",
        )
        intersection_points_gdf = gpd.overlay(
            self.boundaries,
            gpd.GeoDataFrame(geometry=[buffered_point.boundary], crs=WEB_MERCATOR),
            how="intersection",
            keep_geom_type=False,
        )
        intersection_points = []

        # Resultant geometries should either be Point or MultiPoint
        for geometry in intersection_points_gdf.geometry:
            if geometry.geom_type == "Point":
                intersection_points.append(geometry)
            else:
                intersection_points.extend([Point(p.x, p.y) for p in geometry.geoms])

        # For each of the intersection points we want to find the angle formed between the buffered point
        # which divides it into regions.
        angles = []
        for p in intersection_points:
            v = np.array([p.x - point.x, p.y - point.y])
            angles.append((np.arctan2(v[1], v[0]) + 2 * np.pi) % (2 * np.pi))
        angles.sort()

        # We now need to form the regions of the circle that are created by the intersection points
        # represented as ranges of angles. N intersections results in N+1 ranges.
        valid_regions = []
        invalid_regions = []

        if len(angles) == 0:
            valid_regions.append((0.0, 2 * np.pi))
        else:
            for i in range(len(angles)):
                region = (angles[i], angles[(i + 1) % len(angles)])

                # Test the region by casting a ray from the point to the midpoint of the region
                # Odd number of intersections means the region is invalid and vice versa
                # <technically its opposite to the region the point is in but by assumption this is fine>
                if region[1] < region[0]:
                    # This means the region wraps around the circle
                    # Break it into two regions and add them separately
                    region_A = (region[0], 2 * np.pi)
                    region_B = (0.0, region[1])
                    valid_regions.append(region_A) if self._raycast_intersections(
                        region_A, intersection_boundaries_gdf, point, distance
                    ) % 2 == 0 else invalid_regions.append(region_A)
                    valid_regions.append(region_B) if self._raycast_intersections(
                        region_B, intersection_boundaries_gdf, point, distance
                    ) % 2 == 0 else invalid_regions.append(region_B)
                else:
                    valid_regions.append(region) if self._raycast_intersections(
                        region, intersection_boundaries_gdf, point, distance
                    ) % 2 == 0 else invalid_regions.append(region)

        return (valid_regions, invalid_regions)

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

                for line in self.boundaries.geometry:
                    # Generate the a positive, negative and no offset on either side of the geom
                    lines = [
                        line,
                        offset_curve(line, buffer),
                        offset_curve(line, -buffer),
                    ]
                    candidates: Point = []

                    for l in lines:
                        try:
                            candidates.append(nearest_points(l, pt)[0])
                        except ValueError:
                            # Skip empty geometries
                            pass

                    # Get the furthest point out of all candidate points as we want to buffer into valid areas
                    candidate = max(candidates, key=lambda p: p.distance(pt))
                    dist = candidate.distance(pt)

                    if dist < closest_distance:
                        closest_pt = candidate
                        closest_distance = dist

                print(f"Replacing {pt} with {closest_pt}")
                trajectory.geometry.iloc[i] = closest_pt

        return trajectory.drop(columns=["valid"])
