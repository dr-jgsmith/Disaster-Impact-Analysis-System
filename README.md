# Disaster Impact Analysis System (DIAS)

## Overview
The Disaster Impact Analysis System (DIAS) is open source  software that can be used to simulate the impacts of reoccurring flood events on real estate prices and assess flood mitigation strategies using a multi-criteria decision methods. DIAS implements a number of methods for representing the connectivity of urban spaces that can be used to model hydrologic events such as flood and storm surge events. DIAS also includes methods for analyzing mitigation strategies, along with strategy-ranking methods using multi-criteria Q-analysis (MCQA I & II).  DIAS is written in Python 3.6. and makes heavy use of [Numpy](http://www.numpy.org/) and the [Numba](https://numba.pydata.org/) JIT compiler to achieve near C performance for computations involving large sparse matrices.  

## Table of Contents
  * [Installation](#installation)
    + [Clone or download the repository](#clone-or-download-the-repository)
    + [cd and Pip](#cd-and-pip)
  * [Workflow](#workflow)
  * [Importing Data and Setting Attributes](#importing-data-and-setting-attributes)
  * [Representation](#representation)
  * [Simulation](#simulation)
  * [Visualization](#visualization)
  * [Evaluation](#evaluation)
  * [Exporting Results](#exporting-results)

## Installation
### Clone or download the repository

    $ git clone https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System


    $ virtualenv mydias
Mac OS/Linux

     $ source mydias/bin/activate
Windows

    $ mydias\Scripts\activate

### CD and Pip

    $ cd /path/to/your/DIAS
    $ pip install .
The setup.py file will install the required dependencies. 

## Workflow
DIAS is designed primarily to process attribute data associated with [ESRI ArcGIS](https://www.esri.com/en-us/arcgis/about-arcgis/overview) or [QGIS](https://qgis.org/en/site/). These attribute files are typically stored as dBase files ending with the .dbf extension and provide additional data for analyzing spatial information. DIAS consumes and enriches these attribute files.

**![](https://lh6.googleusercontent.com/NbvhAT5cnDjooWlpmdMkQWhdkGLU28BgJIZsSz6ulYNlfMORFYP0a87WSdSBs3ASl6QlM9924YUka26bSRviragoS0RDt5Vcfm9o4hgpfBPaUpMB-QB12Pcm7Zx3shIDiJSjOo2tjII)**
The GIS analyst or user can load a file into DIAS, create a base layer representation of connectivity, simulate dynamic behavior of system attributes (e.g. changes in land value), and export simulation results for visualization in ArcGIS, QGIS or another spatial analytics platform.

Currently, there are two basic ways for using DIAS:

 1. Building Models (representations)
 2.  Simulating model behavior
 
 The `run_model()` function provides a wrapper for building and simulating model behavior. This is the most common use case, and all arguments can be passed to this single function. In some cases, users might want more control over both the build and simulation processes. In this case, DIAS lets users operate the build process separate from the simulation, this allows for models to be saved for later use. This is accomplished by passing **all of the required parameters** for `run_model()` to the `build_base_model()` and `simulate_base_model()`.

 It is assumed that users will run DIAS using [Jupyter Notebook](http://jupyter.org/). However, DIAS can be wrapped within Flask web app and run as a data enrichment service for analysts and planners.
 
## Importing Data and Setting Attributes
DIAS can import and read both `.dbf` and `.csv` file formats. Typically, these data contain at a minimum, an ID such as parcel ID, longitude and latitude, land value, and the value of any structure existing on that land. Additional, attributes can be included such as ownership, flood insurance rates, and owner occupancy, but are unnecessary for our purposes here.

First we begin by importing the base_model and simulate_model modules. Next we need to reference the file (.dbf or .csv) to be passed to the `build_base_model()` function (or `run_model()`).

```python  
from dias.scripts.base_model import * 
from dias scripts.simulate_model import *    

# Begin with dBase file commonly associated with attribute data for annotating layers in a GIS  
file = "C:\\PATH\\TO\\MY\\DBFILE.dbf"   
```
> **Note:** in some cases you might need to collect elevation data, but most often you will need to use a pre-existing file that contains elevations and parcel ids. This is a locally stored `.csv` file that can be called to get parcel elevation data.

```python  
 elevations = "C:\\PATH\\TO\\MY\\ELEVATIONS_FILE.csv" 
```
If you do not have an elevations file to reference, you will need to generate a listing of elevations for each parcel in the data set. To do this, simply supply a  [Google Maps API key](https://developers.google.com/maps/documentation/embed/get-api-key).
```python
 map_key='GOOGLE_MAPS_API_KEY' 
```
The map_key can be passed to either the `run_model(map_key=map_key)` or `build_base_model(map_key=map_key)` functions to . This will build an elevation model using the parcel ids and the geo-coordinates contained in the data file. This will also save the elevations in a new file named `elevations.csv` that can be referenced in future builds. 

Before you can run or build a model using your data, the field names for `latitude`, `longitude` and `parcel id` will need to be defined and supplied. It is also assumed that you will supply references to real estate values. 

```python 
# Define field_names 
lat = 'Lat'
lon = 'Long'
parcel_field = 'PARCELATT'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'
map_key='GOOGLE_MAPS_API_KEY' # optional
```
In addition to the parameter names, values need to be set to initialize the model. 
```python  
# Minimum and Maximum Impact Value 
impact_range = (3, 14)   
time_step = 25  
iterations = 500  
impact_multiplier = 0.8
```
Here we define an `impact_range` and `max_impact` value that represent the range of events. Next we define a `time_step` value that represents the number of compound computations, in the case years.  The number of `iterations` refers to the number of observations for computing a statistical average, and the `impact_multiplier` refers to the full impact loss potential of property. In other words, this is the extreme limit of loss.

## Representation
Now that we have defined some of the fields and initial parameters we can build the structural representation of the system. The model can then be used to simulate system dynamics. First we pass the files and parameters to the `build_base_model()` function. These include a reference to the primary input file, an elevations file, the latitude and longitude fields, a max_impact value and the impact_multiplier.
```python  
model = build_base_model(file, elevations, lat, lon, max_impact, impact_multiplier)   
```
The `build_base_model()` produces a structural representation based on the elevation and connectivity between parcels. The method employs both geodesic or Euclidean distance measures.  The returned data include a connectivity matrix (numpy array), an array of elevations, set of impact zones and an object  reference to the model.

The 'impact zones' are computed by calculating the connected components for each threshold (slice) between a minimum and maximum value.  Each  component represent zones in which water at each slice is free to move from one place to another without obstruction.  If a parcel is above the slicing threshold then the parcel acts as an obstruction. 

An example output below provides a representation of flood an inundation threats given both parcel elevations and proximity. Here elevation is set at 12 feet, so any connected parcel beneath 12 feet in elevation is at risk of flooding.

**![Inundation Zones 12 feet - Carto Map](https://lh6.googleusercontent.com/bgwiYPBtKZjzABEORIZys7Ar4oUn5SKk57UZR05AfW10xD2K9UfTyj6VzbPcT-rKCQX0SMSbbHtGIKoX2Zq4r_4v4zLWoNbg_Yt3tRcJ2OP71cVc9kYv4Ot0qsQc5fWjbBE1nNxdoBU)**

## Simulation
System behavior can be simulated by calling the `simulate_base_model()` function. The function takes a reference to the model object, along with the fields that you want to evaluate ( building and land values), iterations and time step.  
```python 
sim = simulate_base_model(model, building_value_field, land_value_field, impact_range, iterations, time_step, output_file_name)
```
The simulation runs each step of the model based on the number of iterations defined. Higher number of iterations offer a more accurate statistical average because we use a randomizer function to randomize a set of values based on a particular distribution. 

## Visualization

## Evaluation

## Exporting Results
Results are automatically stored as 



