from abc import ABC, abstractmethod
import geopandas as gpd  # type: ignore

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps


class SampleMethod(ABC):
    COLOR = "red"  # Default colour for plotting the method

    @abstractmethod
    def _sample_angle(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def _sample_distance(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def privatise_trajectory(
        self, gdf: gpd.GeoDataFrame, eps: float, delta: float
    ) -> gpd.GeoDataFrame:
        pass
