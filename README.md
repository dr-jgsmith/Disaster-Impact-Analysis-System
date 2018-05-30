# Disaster Impact Analysis System (DIAS)

The Disaster Impact Analysis System (DIAS) is open source  software that can be used to simulate the impacts of reoccurring flood events on real estate prices and assess flood mitigation strategies using a multi-criteria decision methods. DIAS implements a number of methods for representing the connectivity of urban spaces that can be used to model hydrologic events such as flood and storm surge events. DIAS also includes methods for analyzing mitigation strategies, along with strategy-ranking methods using multi-criteria Q-analysis (MCQA I & II).  DIAS is written in Python 3.6. and makes heavy use of [Numpy](http://www.numpy.org/) and the [Numba](https://numba.pydata.org/) JIT compiler to achieve near C performance for computations involving large sparse matrices.  

## Table of Contents
1. [Installation](#Installation)
2. [Worflow](#Workflow)
3. [Importing Data](#Importing Data)
4. [Representation](#Representation)
5. [Simulation](#Simulation)
6. [Visualization](#Visualization)
7. [Evaluation](#Evaluation)
8. [Exporting Results](#Exporting Results)


## Installation
### Clone or download the repository

    $ git clone https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System

### Create a virtual environment 

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

**![](https://lh5.googleusercontent.com/9Ap-AYx1UDZNUp7itb5hcvDCXRrw1PXxR1RXend6tKE9Ott2YUljhCuJGfHu7pXBFQmAvQITydZS14JkB8557NIICIxFolllaO97X2_hFqOLqiXW5wwdSqx-ydWvSr3KjU88gg8lIuo)**

## Importing Data and Setting Attributes
Imported data is assumed to be in some type of tabular form and contain at a minimum, an ID such as parcel ID, longitude and latitude, land value, and the value of any structure existing on that land. Additional, attributes can be included such as ownership, flood insurance rates, and owner occupancy.

```python  
from dias.scripts.flood_model import *     

# Begin with dBase file commonly associated with attribute data for annotating layers in a GIS  
file = "C:\\PATH\\TO\\MY\\DBFILE.dbf"  

# Open the file and intitialize the flood model class  
db = flood_model(file, 'GOOGLE_MAPS_API_KEY')   
```

When initializing the flood_model class, the `openfile()` method in the `processdbf.py` module is called to process the referenced file. This reads the file and converts the data to a column format as a list of tuples. Each tuple contains a column name and a list of values. If a parcel has no geo-location data, the parcel is excluded from the analysis.

The `flood_model()` class also **requires** a [Google Maps API key](https://developers.google.com/maps/documentation/embed/get-api-key) (for now). This is used to collect elevation data for each parcel, if there is no elevation attributes included in the data set. 

Once the file and Maps API key are passed to the `flood_model()` class, it is time to define the data fields we want to use as well as any new fields we might create. For our purposes here, we will focus on only a few fields.

```python  
lat = 'Lat'
lon = 'Long'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'
parcel_field = 'PARCELATT'
```
Next define a new field for storing elevation data.

```python  
# Define a new field, Elevation  
elevation_field = 'Elevation'  
```

## Importing Data

## Representation

## Simulation

## Visualization

## Evaluatation

## Exporting Results




