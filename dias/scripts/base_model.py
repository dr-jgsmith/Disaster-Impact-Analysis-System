import statistics
from dias.storage.processdbf import *
from dias.core.hyper_graph import *
from scipy.spatial import distance
import googlemaps


def calc_elevation(filename, elevation_file, latfield, lonfield, map_key=None):
    """
    Compute the elevation values for each parcel. User can supply a file with parcel ids and elevations,
    or provide reference to a Google Maps API
    :param filename: dbf file type
    :param map_key: requires a google maps api key
    :param latfield: latitude
    :param lonfield: longitude
    :return: a list of elevations
    """
    try:
        if map_key:
            file = processdbf(filename)
            file.openfile()

            gmaps = googlemaps.Client(key=map_key)
            elevations = []

            lat_index = file.get_column(latfield)
            lon_index = file.get_column(lonfield)

            tmp = []
            e = []
            for i in range(len(lat_index[1])):
                # iterate through and check for NaN values.
                try:
                    tmp.append((float(lat_index[1][i]), float(lon_index[1][i])))
                # if value is NaN append the row index value to the e list
                except ValueError:
                    e.append(i)
                    pass

                if len(tmp) > 499:  # API Rate Limit 2500 queries per day | 520 points per query
                    geocode_result = gmaps.elevation(tmp)
                    for k in geocode_result:
                        ele = k['elevation']
                        # convert to feet
                        ele = ele * 3.28084
                        elevations.append(float(ele))
                    tmp = []
                else:
                    pass

            if len(tmp) > 0:
                geocode_result = gmaps.elevation(tmp)
                for l in geocode_result:
                    ele = l['elevation']
                    # convert to feet
                    ele = ele * 3.28084
                    elevations.append(float(ele))
            else:
                pass
            # remove rows with NaN values
            cnt = 0
            for i in e:
                dex = i + 1
                dex = dex - cnt
                del file.output[dex]
                cnt = cnt + 1

            file.add_column('Elevation', elevations)
            return file
        else:
            ele = processdbf(elevation_file)
            ele.open_csv()

            # elevations = ele.get_column("Elevation")
            parcels = ele.get_column('PARCELATT')

            file = processdbf(filename)
            file.openfile()

            data = synch_files(file, ele, parcels)
            # data.add_column('Elevation', elevations[1])
            return data
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


@jit
def synch_files(infile, outfile, column):
    """
    :param infile: 
    :param outfile: 
    :param column: 
    :return: 
    """
    for i in column[1]:
        row = infile.get_row(column[0], i)
        outfile.add_row(row)
    return outfile


def construct_adjacency(file, latfield, lonfield):
    """
    Construct a representation of system connectivity given elevation and proximity.

    :param file: object reference
    :param latfield: float latitude
    :param lonfield: float longitude
    :return: numpy incidence matrix 
    """
    try:
        lat_index = file.get_column(latfield)
        lon_index = file.get_column(lonfield)

        point_list = np.array([(lat_index[1][i], lon_index[1][i]) for i in range(len(lat_index[1]))])
        d = distance.cdist(point_list, point_list, 'euclidean')
        normed = d * 100
        proximity = max(normed[0]) / (max(normed[0]) * 7)

        incident = ['x']
        for j in normed:
            k = np.piecewise(j, [j <= proximity, j > proximity], [1, 0])
            incident.append(k)

        incident = np.vstack(incident[1:])
        elevations = file.get_column('Elevation')
        felevations = [float(i) for i in elevations[1]]

        return incident, felevations, file

    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def elevation_model(input_file, elevation_file, lat, lon, maps_key=None):
    """
    The elevation model can be used to construct a base layer model for computing
    :param input_file: Takes either a .dbf or .csv
    :param lat: latitude value
    :param lon: longitude value
    :param maps_key: string
    :return: tuple with connectivity matrix and vector of elevations 
    """
    print("Getting elevation data...")
    file = calc_elevation(input_file, elevation_file, lat, lon, map_key=maps_key)
    # Compute connectivity of the spatial system
    print("Creating connectivity matrix...")
    connectivity = construct_adjacency(file, lat, lon)
    # Connectivity includes an adjacency matrix, elevation vector and object reference to the data
    return connectivity


@jit
def impact_model(connectivity_matrix, theta):
    """
    Compute the impact of an event given the elevation and connectivity of a given parcel.
    Iterates over all possible impact scenarios and computes the impact of that scenario on real estate values
    :param connectivity_matrix: tuple consisting of an adjacency matrix (numpy) and elevation vector
    :param theta: int or float
    :return: binary array and object reference to data
    """
    print("Generating impact model...")
    elevation = np.array(connectivity_matrix[1])
    zones = ['x']
    for i in range(theta):
        # Given the threshold parameter retain only those values that are above the threshold
        elevation_vector = (elevation <= i).astype(float)
        # Compute pattern on the array
        pattern = computePattern(elevation_vector, connectivity_matrix[0])
        # Compute pattern on the shared-face matrix by taking the transpose of the original shared-face matrix
        pT = pattern.transpose() * elevation_vector
        # Generate a sparse representation
        edges = sparse_graph(pT, range(len(pattern)), 0)
        # Capture connected components as connected risk zones
        classes = compute_classes(edges)
        # Extract zones of risk from the collection of components
        zone_data = np.zeros(len(elevation))
        for j in range(len(classes)):
            for k in classes[j]:
                zone_data[k] = j + 1
        zones.append(zone_data)
        name = "Impact_Zones_" + str(i)
        connectivity_matrix[2].add_column(name, zone_data)
    # return the zones and a reference to the data object
    return zones[1:], connectivity_matrix[2]


@jit
def binarize_zones(zones):
    """
    Takes an list of arrays for each layer and binarizes all non-zero elements
    :param zones: list of layers | list of numpy arrays
    :return: list of binary numpy arrays 
    """
    layers = ['x']
    for i in range(len(zones)):
        zone_vector = np.zeros(len(zones[i]))
        for j in range(len(zone_vector)):
            if zones[i][j] > 0.0:
                zone_vector[j] = 1.0
            else:
                pass
        layers.append(zone_vector)
    return layers[1:]


@jit
def loss_function_Z(elevation_pattern, loss_percent, layer, theta):
    """
    Calculates impact parameter given impact intensity.
    :param elevation_pattern: numpy array | zeros for excluded values, raw elevation in feet
    :param loss_percent: int - a constant value
    :param layer: numpy array - binarized set of impact zones
    :param theta: int - a constant - slicing value
    :return: numpy array - impact multiplier based on elevation and  
    """
    elevations = computePattern(elevation_pattern, layer)
    mask = elevations != 0.0
    if len(elevations[mask]) > 0:
        elevations[mask] = theta - elevations[mask]
        impact_multiplier = loss_percent * np.absolute(elevations)
    else:
        impact_multiplier = np.zeros(len(elevations))
    return impact_multiplier


@jit
def compute_impact_intensity(elevations, layers, bloss_fun, data_object):
    """
    :param elevations: list or numpy array of elevations
    :param layers: binarized vector of impact zones
    :param bloss_fun: loss function value
    :param data_object: object reference to data 
    :return: vector of data and vector of means
    """
    data = ['x']
    means = ['x']
    for i in range(len(layers)):
        field_name = "Intensity_" + str(i)
        # Compute impact
        impact_multiplier = loss_function_Z(elevations, bloss_fun, layers[i], i)
        # Compute and retain means
        means.append(statistics.mean(impact_multiplier))
        data.append(impact_multiplier)
        data_object.add_column(field_name, impact_multiplier)
    return data[1:], means[1:]


@jit
def build_base_model(input_file, elevation_file, lat, lon, max_impact, perc_loss, map_key=None):
    """
    Use this function to build a base model or representation of the system.
    The base model can be saved and used by the simulate_model function.
    :param input_file: .dbf or .csv file path
    :param elevation_file: .csv file path
    :param lat: string
    :param lon: string
    :param max_impact: int
    :param perc_loss: float
    :param map_key: Google Maps API Key **optional
    :return: tuple | impact layers, zone vector, elevations, impact matrices, object reference
    """
    # Call elevation model to construct the connectivity model
    data = elevation_model(input_file, elevation_file, lat, lon, maps_key=map_key)
    # Call the impact model | this will generate a series of impact layers given an impact threshold
    # This returns a vector of labels connected components
    zones = impact_model(data, max_impact)
    # binarize zones | if a zone (int greater the 0) is present, then value is equal to 1.
    layers = binarize_zones(zones[0])
    # using the binarized pattern, compute the intensity of an impact
    impact_intensities = compute_impact_intensity(data[1], layers, perc_loss, zones[1])
    # return binarized layers, zone vector, elevations, impact matrices, model object reference
    return layers, zones[0], data[1], impact_intensities, zones[1]