from abc import ABC, abstractmethod
import geopandas as gpd  # type: ignore


class SampleMethod(ABC):
    @abstractmethod
    def _sample_angle(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _sample_distance(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float
    ) -> gpd.GeoDataFrame:
        pass
