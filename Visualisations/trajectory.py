import pandas as pd
import os
from pathlib import Path
import geopandas as gpd  # type: ignore

from commands.error_preset import command_error_preset
from commands.error_varying_epsilon import command_error_varying_epsilon
from commands.error_varying_delta import command_error_varying_delta
from commands.privatise import command_privatise
from commands.privatise_preset import command_privatise_preset
from commands.privatise_varying_delta import command_privatise_varying_delta
from commands.privatise_varying_epsilon import command_privatise_varying_epsilon

from common import WEB_MERCATOR, WGS_84

# Constants pointing to the dataset and coastline shapefile
DATASET_PATH = os.path.join(Path(__file__).parent.parent, "AIS_2019_01_01.csv")
FLORIDA_COASTLINE = "Shapefiles/florida_coastline.shp"

# Drop these columns from the dataset as they are not needed
EXCESS_COLUMNS = [
    "SOG",
    "COG",
    "Heading",
    "IMO",
    "Status",
    "Length",
    "Width",
    "Draft",
    "Cargo",
    "CallSign",
    "VesselType",
    "TransceiverClass",
]
MODES = {
    "q": "Quit",
    "p": "Privatise",
    "pd": "Privatise, varying delta",
    "pe": "Privatise, varying epsilon",
    "pp": "Privatise, preset",
    "ep": "Error Comparison, preset",
    "ee": "Error Comparison, varying epsilon",
    "ed": "Error Comparison, varying delta",
}


class Privatiser:
    gdf: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame

    def __init__(self) -> None:
        self.gdf = self.load_initial_data()
        self.florida_coastline = self.load_florida_coastline()

    def load_initial_data(self) -> gpd.GeoDataFrame:
        """
        Load the dataset into python and clean it by removing excess columns and
        reprojecting to web mercator
        """

        # Load the dataset and drop the excess columns
        df = pd.read_csv(DATASET_PATH)
        df.drop(
            labels=EXCESS_COLUMNS,
            axis=1,
            inplace=True,
        )

        # Convert to GeoDataFrame with web mercator projection of lat, lon
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LON, df.LAT),
            crs=WGS_84,
        )

        return gdf.to_crs(WEB_MERCATOR)

    def load_florida_coastline(self) -> gpd.GeoDataFrame:
        """
        Load the florida coastline shapefile and reproject to web mercator
        """

        florida_coastline = gpd.read_file(FLORIDA_COASTLINE)

        return florida_coastline.to_crs(WEB_MERCATOR)

    def run_command(self):
        """
        Determine which command to run to select an experiment to run
        """

        commands = "\n".join([f"\t- {v} ({k})" for k, v in MODES.items()])
        mode = input(f"Enter a command:\n {commands}\n")

        if mode == "q":
            exit()
        elif mode == "p":
            command_privatise(self.gdf, self.florida_coastline)
        elif mode == "pd":
            command_privatise_varying_delta(self.gdf, self.florida_coastline)
        elif mode == "pe":
            command_privatise_varying_epsilon(self.gdf, self.florida_coastline)
        elif mode == "pp":
            command_privatise_preset(self.gdf, self.florida_coastline)
        elif mode == "ep":
            command_error_preset(self.gdf, self.florida_coastline)
        elif mode == "ee":
            command_error_varying_epsilon(self.gdf, self.florida_coastline)
        elif mode == "ed":
            command_error_varying_delta(self.gdf, self.florida_coastline)
        else:
            print("Invalid command")


if __name__ == "__main__":
    privatiser = Privatiser()

    while True:
        privatiser.run_command()
