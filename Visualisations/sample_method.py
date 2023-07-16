from abc import ABC, abstractmethod
import geopandas as gpd  # type: ignore
import numpy as np
from shapely.geometry import Point  # type: ignore

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps


class SampleMethod(ABC):
    @abstractmethod
    def _sample_angle(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _sample_distance(self, *args, **kwargs) -> float:
        pass

    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float
    ) -> gpd.GeoDataFrame:
        """
        Privatises the trajectory of a ship using our developed method.
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
