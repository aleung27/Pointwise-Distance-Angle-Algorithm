import numpy as np
from scipy.stats import rv_continuous  # type: ignore
import geopandas as gpd  # type: ignore
from shapely.geometry import Point  # type: ignore

from sample_method import SampleMethod
from constants import WEB_MERCATOR


class SampleDistanceDirection(SampleMethod):
    """
    Contains the Sample Distance and Direction (SDD) sampling functions.
    """

    COLOR = "purple"
    NAME = "SDD"

    def _sample_angle(self, eps: float, theta: float) -> float:
        """
        Samples an angle theta according to the SDD method given by
        Pr(θ) = C*exp(-ε*|α_i-θ|/16π)
        Here we have:
        Δu = 2π, u(α, θ) = |α-θ| where θ, α ∈ [0, 2π]
        """
        exp_factor = 16 * np.pi / eps

        def inverse_cdf(X: float, eps: float, theta: float) -> float:
            # Constant from PDF
            C = (
                2
                - np.exp(-(2 * np.pi - theta) / exp_factor)
                - np.exp(-theta / exp_factor)
            )

            # This is the inverted piece for when angles are in the range [0, theta]
            piecewise_A = (
                theta + np.log(C * X + np.exp(-theta / exp_factor)) * exp_factor
            )

            # This is the inverted piece for when angles are in the range [theta, 2pi]
            piecewise_B = (
                theta - np.log(-C * X + 2 - np.exp(-theta / exp_factor)) * exp_factor
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

    def _sample_distance(self, eps: float, r: float, M: float) -> float:
        """
        Samples a distance according to the SDD method given by the following
        Pr(r) = C*exp(-ε*|x-r|/8M)
        Here M refers to the max distance between any two points in the trajectory:
        If we have that r >= M, then we have:
        1/C = 8M(e^(-(r-M)ε/8M) - e^(-εr/8M))/ε
        If instead we have that r < M, then we have:
        1/C = 8M(2 - e^(-(M-r)ε/8M) - e^(-εr/8M))/ε

        Here we subclass the rv_continuous class from scipy.stats to generate
        the distribution. This allows us to use the rvs() method to sample from
        the distribution rather than analytically inverting the CDF.
        """

        class DistanceDistribution(rv_continuous):
            def __init__(self, eps: float, r: float, M: float, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.eps = eps
                self.r = r
                self.M = M

            def _pdf(self, x, *args):
                exp_factor = 8 * self.M / self.eps

                return np.where(
                    self.r >= self.M,
                    np.exp(-abs(x - self.r) / exp_factor)
                    / (
                        exp_factor
                        * (
                            np.exp(-(self.r - self.M) / exp_factor)
                            - np.exp(-self.r / exp_factor)
                        )
                    ),
                    np.exp(-abs(x - self.r) / exp_factor)
                    / (
                        exp_factor
                        * (
                            2
                            - np.exp(-self.r / exp_factor)
                            - np.exp(-(self.M - self.r) / exp_factor)
                        )
                    ),
                )

        distribution = DistanceDistribution(eps, r, M, a=0, b=M)

        return distribution.rvs()

    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float, delta: float
    ) -> gpd.GeoDataFrame:
        """
        Privatises the given trajectory using the SDD method.
        """

        def euclidean_distance(a: Point, b: Point) -> float:
            return float(np.linalg.norm(np.array([a.x - b.x, a.y - b.y])))

        # Privatised route starts and ends at the same location as the real route
        privatised = {
            "geometry": [gdf.geometry.iloc[0]],
        }

        # Get the maximum distance between any two points in the trajectory
        M: float = np.max(
            [
                euclidean_distance(gdf.geometry.iloc[i], gdf.geometry.iloc[i - 1])
                for i in range(1, len(gdf))
            ]
        )

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
            distance = float(np.linalg.norm(v))
            angle = (np.arctan2(v[1], v[0]) + 2 * np.pi) % (2 * np.pi)

            i = 0
            while True:
                # Sample a distance and angle from the given distributions
                sampled_distance = self._sample_distance(eps, distance, M)
                sampled_angle = self._sample_angle(eps, angle)

                new_point = Point(
                    last_estimate.x + sampled_distance * np.cos(sampled_angle),
                    last_estimate.y + sampled_distance * np.sin(sampled_angle),
                )

                if (
                    euclidean_distance(gdf.geometry.iloc[-1], new_point)
                    < (len(gdf) - i) * M
                    or i > 100000000  # Break after 100 million iterations
                ):
                    privatised["geometry"].append(new_point)
                    break

                i += 1

        privatised["geometry"].append(gdf.geometry.iloc[-1])
        return gpd.GeoDataFrame(privatised, geometry="geometry", crs=WEB_MERCATOR)
