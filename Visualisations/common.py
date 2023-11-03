from matplotlib.ticker import ScalarFormatter  # type: ignore

WEB_MERCATOR = "EPSG:3857"  # Standard flat projection for web maps
WGS_84 = "EPSG:4326"  # Projection for latitude and longitude

# Preset variables for use in experimentation
PRESET_MMSIS = [
    258288000,
    316038559,
    367637340,
    368011140,
]
PRESET_EPSILONS = [0.1, 0.5, 1, 2, 5, 10]
PRESET_DELTAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
TEST_ITERATIONS = 100


class CustomScalarFormatter(ScalarFormatter):
    """
    Custom scalar formatter to format the axes to be 4 sig figs with sci notation
    """

    def _set_format(self):
        self.format = "%#.4g"
