# Disaster Impact Analysis System (DIAS)

The Disaster Impact Analysis System (DIAS) is open source  software that can be used to simulate the impacts of reoccurring flood events on real estate prices and assess flood mitigation strategies using a multi-criteria decision methods. DIAS implements a number of methods for representing the connectivity of urban spaces that can be used to model hydrologic events such as flood and storm surge events. DIAS also includes methods for analyzing mitigation strategies, along with strategy-ranking methods using multi-criteria Q-analysis (MCQA I & II).  DIAS is written in Python 3.6. and makes heavy use of [Numpy](http://www.numpy.org/) and the [Numba](https://numba.pydata.org/) JIT compiler to achieve near C performance for computations involving large sparse matrices.  

## Table of Contents

[TOC]

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
DIAS is designed primarily to process attribute data associated with ESRI ArcGIS or QGIS. These attribute files are typically stored as dBase files ending with the .dbf extension.

## Importing Data

## Define the Base

## Define Simulation Parameters

## Visualization

## Evaluate Mitigation Strategies

## Exporting Results




