from dias.scripts.simulate_model import *

"""
This begins the example use case of value_impact_analysis

"""
# Begin with dBase file commonly associated with attribute data for annotating layers in a GIS
file = "/Users/justinsmith/Desktop/Data/GHC_DATA/AB_HOQ_DATA.dbf"
# Open the file and intitialize the flood model class

elevations = "/Users/justinsmith/Desktop/Data/GHC_DATA/elevations.csv"


# Define the column or id names
# These will be hard coded and should not be allowed to change
lat = 'Lat'
lon = 'Long'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'
parcel_field = 'PARCELATT'
impact_field = "Impact_Zones_12"
output = 'output.csv'
impact_range = (3, 14)
max_impact = 14
time_step = 25
iterations = 500
impact_multiplier = 0.8


model = build_base_model(file, elevations, lat, lon, max_impact, impact_multiplier)
print(model)

sim = simulate_base_model(model, building_value_field, land_value_field, impact_range, iterations, time_step, output)

zones = impact_by_zone(sim, parcel_field, impact_field)
print(zones)

base = model_growth(model[4], building_value_field, land_value_field, growth_rate=0.04, time_step=time_step)
print(base[1])

# run_model(file, elevations, lat, lon,  building_value_field, land_value_field, parcel_field, impact_field, impact_range, max_impact, impact_multiplier, iterations, time_step, output_file_name)
