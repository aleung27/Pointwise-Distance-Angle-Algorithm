from sample_method import SampleMethod

from typing import List, Tuple
import numpy as np
from scipy.stats import rv_continuous  # type: ignore


class AvoidantSamplingMethod(SampleMethod):
    """
    Contains the Avoidant Sampling Method sampling functions. This method allows sampling
    an angle whilst specifying multiple disjoint regions to avoid.
    """

    # Regions are of the form [lower, upper]. These are sorted and disjoint.
    avoidance_regions: List[Tuple[float, float]]
    allowed_regions: List[Tuple[float, float]]

    def __init__(self, avoidance_regions: List[Tuple[float, float]]) -> None:
        def form_allowed_regions():
            self.allowed_regions = []

            if len(avoidance_regions) == 0:
                self.allowed_regions.append((0.0, 2 * np.pi))
                return

            for i, region in enumerate(avoidance_regions):
                if not np.isclose(region[0], 0.0):
                    self.allowed_regions.append(
                        (avoidance_regions[i - 1][1] if i else 0.0, region[0])
                    )

            if not np.isclose(avoidance_regions[-1][1], 2 * np.pi):
                self.allowed_regions.append((avoidance_regions[-1][1], 2 * np.pi))

        self.avoidance_regions = avoidance_regions
        form_allowed_regions()

    def _sample_angle(self, eps: float, theta: float) -> float:
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
                avoidance_regions: List[Tuple[float, float]],
                allowed_regions: List[Tuple[float, float]],
                *args,
                **kwargs
            ) -> None:
                super().__init__(*args, **kwargs)
                self.eps = eps
                self.avoidance_regions = avoidance_regions
                self.allowed_regions = allowed_regions

            def _pdf(self, x, *args):
                def utility(x: float):
                    """
                    This is the utility function for the angle sampling method.
                    """
                    return (
                        1
                        if np.any([r[0] <= x <= r[1] for r in self.allowed_regions])
                        else 0
                    )

                allowed_region_sum = np.sum(
                    [r[1] - r[0] for r in self.allowed_regions]
                ) * np.exp(self.eps / 2)
                avoidance_region_sum = np.sum(
                    [r[1] - r[0] for r in self.avoidance_regions]
                )

                return np.exp(self.eps * utility(x) / 2) * (
                    1 / (allowed_region_sum + avoidance_region_sum)
                )

        distribution = DistanceDistribution(
            eps=eps,
            avoidance_regions=self.avoidance_regions,
            allowed_regions=self.allowed_regions,
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
