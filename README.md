# Pointwise Distance Angle (PDA) Algorithm

Pointwise Distance Angle (PDA) algorithm is a **differentially private** algorithm for the anonymisation of trajectory datasets. Here, we present an implementation in Python showing a novel approach which retains the utility of a trajectory whilst satisfying the guarantees provided by (ε, δ)-differential privacy. This repo provides a series of commands to demonstrate the performance of PDA against other state of the art methodologies such as Sample Distance and Direction (SDD) proposed by Jiang et al. From our experiments, we determine reduction in errors greater than 70% against existing methodologies with only slight degradations in the privacy guarantees.

## Setup

Perform the following instructions prior to running the program.

### Downloading Depedencies

This project was developed in Python 3.10. A `requirements.txt` file is provided but it is overly exhaustive, being generated from all the site packages on my development environment. As a general guide, the following packages have been leveraged in this project:

- numpy
- scipy
- pandas
- geopandas
- shapely
- python-dtw
- similaritymeasures
- matplotlib

### Downloading NOAA AIS Dataset

Please download the NOAA AIS dataset from [MaringCadastre](https://marinecadastre.gov/ais/) and place the extracted csv file into the root of the folder. The AIS_2019_01_01 dataset was used as part of this thesis.

### Downloading the Florida Coastline shapefiles

Please download the shapefile for geographic boundaries from [OSM](https://mapcruzin.com/free-united-states-shapefiles/free-florida-arcgis-maps-shapefiles.htm). The file needed is `Florida Coastline Shapefile` which should be extracted to a folder called `Shapefiles` underneath the `Visualisations` directory. This shapefile denotes the geographic boundaries of Florida and is used to test postprocessing of trajectories which fall in land.

## Usage

To run the program navigate to the `Visualisations` folder and run:

```[bash]
python3 trajectory.py
```

### Modes

- `Privatise (p)`: Privatises a given ship's trajectory based on a given MMSI, epsilon and delta value. Plots all resultant trajectories on a web mercator map for visual comparison.
- `Privatise, varying delta (pd)`: Privatises a given ship's trajectory against a variety of different delta values and a supplied epsilon value. Plots all resultant trajectories on a 2 x 3 web mercator map for visual comparison of the effect delta has on the privatisation process.
- `Privatise, varying epsilon (pd)`: Privatises a given ship's trajectory against a variety of different epsilon values and a supplied delta value. Plots all resultant trajectories on a 2 x 3 web mercator map for visual comparison of the effect epsilon has on the privatisation process.
- `Privatise, preset (pp)`: Privatises a series of preselected ship trajectories for a given value of epsilon and delta value. Plots the resultant trajectories on a 2 x 2 web mercator map showing the performance of each method.
- `Error, preset (ep)`: Performs N = 100 privatisations for preselected ship trajectories given a value of epsilon and delta value. Calculates the average, minimum and maximum errors according to DTW and DFD and plots them as a bar chart with errors.
- `Errpr, varying delta (ed)`: Performs N = 100 privatisations for a given ship's trajectory against a variety of different delta values and a supplied epsilon value. Calculates the average, minimum and maximum errors according to DTW and DFD and plots them as a line chart with errors.
- `Privatise, varying epsilon (pd)`: Performs N = 100 privatisations for a given ship's trajectory against a variety of different epsilon values and a supplied delta value. Calculates the average, minimum and maximum errors according to DTW and DFD and plots them as a line chart with errors.
