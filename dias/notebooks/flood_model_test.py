from dias.scripts.flood_model import *
import matplotlib.pyplot as plt

"""
This begins the example use case of value_impact_analysis

"""
# Begin with dBase file commonly associated with attribute data for annotating layers in a GIS
file = "C:\\Users\\justi\\Desktop\\Ab_Hoq_Cosi_Parcels_Lat_Long_Points\\Ab_Hoq_Cosi_Parcels_Lat_Long_Critical_Facilities_NEHRP.dbf"
# Open the file and intitialize the flood model class
db = flood_model(file, 'AIzaSyBWEqdC3EjiDzljsSvJi3v6wUWNKQZAl_g')

# Define the column or id names
# These will be hard coded and should not be allowed to change
lat = 'Lat'
lon = 'Long'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'
parcel_field = 'PARCELATT'
# Define a new field, Elevation
elevation_field = 'Elevation'

# Model Parameters can change and these will be used for running simulations and for building some of the base layers.
# Define parameters
# The threshold parameter relates to flood water level in feet
max_impact = 15
impact_range = (5, 14)
# bloss_fun is the percent of value lost on building that has been effected by flood waters.
# This number can be modified. Or could be random within a range based on available data.
building_loss_fun = 0.07
# lloss_fun is the percent of value lost on actual land.
# In a dynamical system, this might be dependent on frequency of events
land_loss_fun = 0.002
# define a set of start points, these are parcels that would likely see the first impacts this allows for points to
# be defined close to bodies of water as well as points where infrastructure fails to accommodate flow.
disturbance_points = ['517100221020', '026800000101']

data = db.build_base_model(parcel_field, building_value_field, land_value_field, lat, lon, elevation_field)

risk_zones = db.build_risk_layers(data, max_impact)
print("Model Constructed...")
print(len(data[1]))

brisk_zones = db.binarize_risk_layers(data[1], risk_zones)
print(brisk_zones)

impact_intensity = db.compute_impact_intensity(data[4][1], brisk_zones, building_loss_fun)
print(impact_intensity)

print('Initializing simulation...')
values = db.run_model_s(data, risk_zones, impact_range, disturbance_points, bloss_value=building_loss_fun, lloss_value=land_loss_fun,
                        ngrowth=0.04, time_step=20)
print("Model Computed...")

new_values = db.run_static_model(data, brisk_zones, bloss_value=building_loss_fun, lloss_value=land_loss_fun)

dynamics = db.run_dynamic_model(data, brisk_zones, impact_range, bloss_value=building_loss_fun, lloss_value=land_loss_fun, multiplier=5, ngrowth=0.05, time_step=50)

cfs = db.get_critical_zones('Crit_Facil', 9)
print(cfs)

facilities = db.get_critical_facilities('Crit_Facil')

output = db.summarize_building_impact(start=0, stop=50)
print(output)