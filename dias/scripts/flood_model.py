import random
import statistics
import numpy as np
from dias.storage.processdbf import *
from dias.core.hyper_graph import *
from scipy.spatial import distance
import googlemaps


class flood_model:

    def __init__(self, filename, maps_key):
        """
        :param filename: requires full path name and file extension
        """
        self.file = processdbf(filename)
        self.file.openfile()
        self.gmaps = googlemaps.Client(key=maps_key)


    def calc_elevation(self, latfield, lonfield, newfield):
        """
        :param lat: float latitude
        :param lon: float longitude
        :return: list of lists
        """
        try:
            elevation = []

            lat_index = self.file.get_column(latfield)
            lon_index = self.file.get_column(lonfield)

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
                    geocode_result = self.gmaps.elevation(tmp)
                    for k in geocode_result:
                        ele = k['elevation']
                        # convert to feet
                        ele = ele * 3.28084
                        elevation.append(ele)
                    tmp = []
                else:
                    pass

            if len(tmp) > 0:
                geocode_result = self.gmaps.elevation(tmp)
                for l in geocode_result:
                    ele = l['elevation']
                    # convert to feet
                    ele = ele * 3.28084
                    elevation.append(ele)
            else:
                pass
            # remove rows with NaN values
            cnt = 0
            for i in e:
                dex = i + 1
                dex = dex - cnt
                del self.file.output[dex]
                cnt = cnt + 1

            self.file.add_column(newfield, elevation)
            return self.file.output

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


    def construct_adjacency(self, latfield, lonfield):
        """
        :param latfield: float latitude
        :param lonfield: float longitude
        :return: numpy incidence matrix
        """
        try:
            lat_index = self.file.get_column(latfield)
            lon_index = self.file.get_column(lonfield)

            point_list = np.array([(lat_index[1][i], lon_index[1][i]) for i in range(len(lat_index[1]))])
            d = distance.cdist(point_list, point_list, 'euclidean')
            normed = d * 100
            proximity = max(normed[0]) / (max(normed[0]) * 7)

            incident = []
            for j in normed:
                k = np.piecewise(j, [j <= proximity, j > proximity], [1, 0])
                incident.append(k)
            incident = np.vstack(incident)

            return incident

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


    def computeImpactZones(self, incident, parcels, elevation_index, threshold):
        """
        :param incident: 
        :param parcel_field: 
        :param elevation_field: 
        :param threshold: 
        :return: 
        """
        # Need to begin with the hyperedges, this case parcels.
        # Each parcel id represents a simplex
        try:
            # Initialize empty sets
            elevations = []
            for i in elevation_index:
                if i > float(threshold):
                    elevations.append(0)
                elif i < float(threshold):
                    # include the value of parcels that could be affected
                    elevations.append(1)
                else:
                    elevations.append(0)

            # Compute the pattern over the incidence matrix representation
            new_matrix = computePattern(tuple(elevations), incident)
            # returns matrix of zeros and float elevation values.
            # values greater than zero means a 1 was present in the incidence matrix.
            # print("Computed pattern on the incidence: ", new_matrix)
            # Run Q-ana the new matrix and threshold value
            edges = sparse_graph(new_matrix, parcels, 0)
            classes = compute_classes(edges)

            zone_data = np.zeros(len(parcels))
            for i in range(len(classes)):
                for j in classes[i]:
                    zone_data[j] = i+1

            field_name = 'Impact_Zones_' + str(threshold)
            # print(zone_data, field_name)
            self.file.add_column(field_name, zone_data)
            # self.file.save_csv('dbf_test.csv')
            return field_name, zone_data

        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Impact Zones')
            pass
        except NameError:
            print('Name Error')
            pass


    def sort_zones(self, impact_point, parcel_list, zone_list):
        """
        :param impact_point: 
        :param parcel_list: 
        :param zone_list: 
        :return: 
        """
        # Sort connected components into vector - indexing method
        zoned = [zone_list[parcel_list.index(i)] for i in impact_point if i is not float]
        zone_vector = np.zeros(len(parcel_list))
        for i in zoned:
            if i == 0.0:
                pass
            else:
                for j in range(len(zone_list)):
                    if i == zone_list[j]:
                        zone_vector[j] = 1.0
                    else:
                        pass

        return zone_vector


    def computeNaturalGrowth(self, property_values, growth_rate):
        """
        :param property_values: 
        :param growth_rate: 
        :return: 
        """
        try:
            np_property_values = np.array(property_values)
            added_value = growth_rate * np_property_values
            new_value = np_property_values + added_value
            return new_value
        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Natural Growth')
            pass
        except NameError:
            print('Name Error')
            pass


    def computeLossFunctionZ(self, elevation_data, loss_percent, theta, zone_vector):
        """
        :param field_name: 
        :param loss_percent: 
        :param theta: 
        :param zone_vector: 
        :return: 
        """
        try:
            elevations = computePattern(elevation_data, zone_vector)
            mask = elevations != 0.0
            elevations[mask] = float(theta) - elevations[mask]
            impact_multiplier = loss_percent * np.absolute(elevations)
            return impact_multiplier

        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Elevation Layer')
            pass
        except NameError:
            print('Name Error')
            pass


    def computeLossFunctionF(self, loss_percent, frequency):
        """
        :param loss_percent: 
        :param frequency: 
        :return: 
        """
        try:
            impact_multiplier = loss_percent * frequency
            return impact_multiplier
        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Loss by Freq.')
            pass
        except NameError:
            print('Name Error')
            pass



    def computeLoss(self, values, impact_multiplier, zone_vector):
        """
        :param values: 
        :param impact_multiplier: 
        :param zone_vector: 
        :return: 
        """
        try:
            bvalues = computePattern(values, zone_vector)
            loss = impact_multiplier * bvalues
            new_val = np.subtract(values, loss)
            return new_val, loss

        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Loss Value Computation')
            pass
        except NameError:
            print('Name Error')
            pass


    def computeImpactCost(self, zone_vector, parcels, bvalues, lvalues, elevations, bloss_fun, lloss_fun, theta, frequency):
        """
        :param zones: list of connected parcel sets
        :param parcel_field: field for parcel ids
        :param bfield: building value field name
        :param lfield: land value field name
        :param impact_point: a list of initial disturbance points
        :param bloss_fun: float loss
        :param lloss_fun: 
        :return: 
        """
        try:

            b_impact_multiplier = self.computeLossFunctionZ(elevations, bloss_fun, theta, zone_vector)
            newb_val = self.computeLoss(bvalues, b_impact_multiplier, zone_vector)

            l_impact_multiplier = self.computeLossFunctionF(lloss_fun, frequency)
            newl_val = self.computeLoss(lvalues, l_impact_multiplier, zone_vector)

            total = np.sum([newb_val[0], newl_val[0]], axis=0)
            tloss = np.sum([newl_val[1], newb_val[1]], axis=0)

            return newb_val[0], newb_val[1], newl_val[0], newl_val[1], total, tloss

        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Compute Impact Cost')
            pass
        except NameError:
            print('Name Error')
            pass


    def build_base_model(self, parcel_field, bvalue_field, lvalue_field, latfield, lonfield, elevation_field):
        """
        :param parcel_field: 
        :param bvalue_field: 
        :param lvalue_field: 
        :param latfield: 
        :param lonfield: 
        :param elevation_field: 
        :return: 
        """
        try:
            print("Building Base Model Layer...")
            self.calc_elevation(latfield, lonfield, elevation_field)
            connectivity_matrix = self.construct_adjacency(latfield, lonfield)
            parcels = self.file.get_column(parcel_field)
            building_values = self.file.get_column(bvalue_field)
            land_values = self.file.get_column(lvalue_field)
            elevations = self.file.get_column(elevation_field)
            return connectivity_matrix, parcels, building_values, land_values, elevations
        except MemoryError:
            print('Memory Error')
            pass
        except RuntimeError:
            print('Runtime Error')
            pass
        except TypeError:
            print('Type Error - Model Constructor')
            pass
        except NameError:
            print('Name Error')
            pass


    def build_risk_layers(self, base_model, max_intensity):
        print('Building Effect-Connected Layers...')
        risk_zones = []
        for i in  range(max_intensity):
            impact_zones = self.computeImpactZones(base_model[0], list(base_model[0][1]), list(base_model[4][1]), i)
            risk_zones.append(impact_zones)
        print('Effected layers computed...')
        return risk_zones


    def binarize_all_zones(self, parcel_list, zone_list):
        zone_vector = np.zeros(len(zone_list))
        for i in range(len(zone_list)):
            if zone_list[i] > 0.0:
                zone_vector[i] = 1.0
            else:
                pass
        return zone_vector


    def binarize_risk_layers(self, parcels, risk_zones):
        brisk_zones = []
        for i in risk_zones:
            zones = self.binarize_all_zones(parcels, i[1])
            brisk_zones.append(zones)
        return brisk_zones


    def run_static_model(self, model_structure, risk_zones,  bloss_value=0.02, lloss_value=0.001):
        """
        :param model_structure: 
        :param impact_range: 
        :param impact_points: 
        :param bloss_value: 
        :param lloss_value: 
        :param ngrowth: 
        :param time_step: 
        :return: 
        """
        print("Running simulation...")
        parcels = model_structure[1]
        building_values = model_structure[2]
        land_values =  model_structure[3]
        elevations = model_structure[4]

        bvalues = [np.array(building_values[1])]
        lvalues = [np.array(land_values[1])]
        tvalues = [np.sum([building_values[1], land_values[1]], axis=0)]

        for i in range(len(risk_zones)):
            intensity = i
            # impact_zones = self.computeImpactZones(connectivity_matrix, parcels[1], elevations[1], intensity)
            values = self.computeImpactCost(risk_zones[i], parcels[1], bvalues[i], lvalues[i], elevations[1], bloss_value, lloss_value, intensity, 1)
            bvalues.append(values[0])
            lvalues.append(values[2])
            tvalues.append(values[4])


        for i in range(len(bvalues)):
            b_field_name = "Build_Value_Impact_Intensity_" + str(i)
            l_field_name = "Land_Value_Impact_Intensity__Year_" + str(i)
            t_field_name = "Total_Value_Impact_Intensity__" + str(i)

            self.file.add_column(b_field_name, bvalues[i])
            self.file.add_column(l_field_name, lvalues[i])
            self.file.add_column(t_field_name, tvalues[i])

        self.file.save_csv('flood_model.csv')
        print("Simulation complete...")
        return bvalues, lvalues, tvalues


    def run_dynamic_model(self, start_values, risk_zones, impact_range, bloss_value, lloss_value, ngrowth=0.04, multiplier=10, time_step=10):

        parcels = start_values[1]
        building_values = start_values[2]
        land_values = start_values[3]
        elevations = start_values[4]

        bvalues = [np.array(building_values[1])]
        lvalues = [np.array(land_values[1])]

        for i in range(time_step):
            bv = []
            lv = []
            tv = []
            event_frequency = random.randint(0, 3)
            if event_frequency > 0:
                intensity = max([self.event_randomizer(impact_range[0], impact_range[1], multiplier) for i in range(3)])
                print(intensity)
                print(risk_zones[intensity])
                values = self.computeImpactCost(risk_zones[intensity], parcels[1], bvalues[i], lvalues[i], elevations[1],
                                                bloss_value, lloss_value, intensity, event_frequency)
                bv.append(values[1])
                lv.append(values[3])

            else:
                bv.append(np.zeros(len(parcels[1])))
                lv.append(np.zeros(len(parcels[1])))

            bval_lost = np.sum(bv, axis=0)
            lval_lost = np.sum(lv, axis=0)

            # check if event occurred. if event occurred pause normal growth rate
            if event_frequency > 0:
                bgvals = self.computeNaturalGrowth(bvalues[i], 0.01)
                lgvals = self.computeNaturalGrowth(lvalues[i], 0.001)
            # compute the normal growth rate
            else:
                bgvals = self.computeNaturalGrowth(bvalues[i], ngrowth)
                lgvals = self.computeNaturalGrowth(lvalues[i], lloss_value)

            bgvals = np.subtract(bgvals, bval_lost)
            lgvals = np.subtract(lgvals, lval_lost)

            bvalues.append(bgvals)
            lvalues.append(lgvals)

        for i in range(len(bvalues)):
            b_field_name = "Total_Build_Value_Year_" + str(i)
            l_field_name = "Total_Land_Value_Year_" + str(i)

            self.file.add_column(b_field_name, bvalues[i])
            self.file.add_column(l_field_name, lvalues[i])

        self.file.save_csv('flood_model.csv')
        print("Simulation complete...")
        return bvalues, lvalues


    def run_model_s(self, model_structure, risk_zones, impact_range, impact_points, bloss_value=0.02,
                    lloss_value=0.001, ngrowth=0.01, time_step=10):
        """
        :param model_structure: 
        :param impact_range: 
        :param impact_points: 
        :param bloss_value: 
        :param lloss_value: 
        :param ngrowth: 
        :param time_step: 
        :return: 
        """
        print("Running simulation...")
        parcels = model_structure[1]
        building_values = model_structure[2]
        land_values = model_structure[3]
        elevations = model_structure[4]

        bvalues = [np.array(building_values[1])]
        lvalues = [np.array(land_values[1])]
        tvalues = [np.sum([building_values[1], land_values[1]], axis=0)]

        for i in range(time_step):
            bv = []
            lv = []
            tv = []

            event_frequency = random.randint(0, 3)

            if event_frequency > 0:
                intensity = max([random.randint(impact_range[0], impact_range[1]) for i in range(event_frequency)])
                # impact_zones = self.computeImpactZones(connectivity_matrix, parcels[1], elevations[1], intensity)
                impact_zones = risk_zones[intensity][1]

                if impact_zones is None:
                    bv.append(np.zeros(len(parcels[1])))
                    lv.append(np.zeros(len(parcels[1])))
                    tv.append(np.zeros(len(parcels[1])))
                else:
                    # Sort connected components into vector - indexing method
                    zone_vector = self.sort_zones(impact_points, parcels[1], impact_zones)
                    values = self.computeImpactCost(zone_vector, parcels[1], bvalues[i], lvalues[i],
                                                    elevations[1], bloss_value, lloss_value, intensity,
                                                    event_frequency)
                    bv.append(values[1])
                    lv.append(values[3])
                    tv.append(values[5])

            # sum over all events for the time-step
            if len(bv) > 0:
                bval_lost = np.sum(bv, axis=0)
                lval_lost = np.sum(lv, axis=0)
                tval_lost = np.sum(tv, axis=0)
            # if no events occurred create an array of zeros
            else:
                bval_lost = np.zeros(len(parcels[1]))
                lval_lost = np.zeros(len(parcels[1]))
                tval_lost = np.zeros(len(parcels[1]))

            # check if event occurred. if event occurred pause normal growth rate
            if event_frequency > 0:
                bgvals = self.computeNaturalGrowth(bvalues[i], 0.0)
                lgvals = self.computeNaturalGrowth(lvalues[i], 0.0)
                tgvals = np.sum([bgvals, lgvals], axis=0)
            # compute the normal growth rate
            else:
                bgvals = self.computeNaturalGrowth(bvalues[i], ngrowth)
                lgvals = self.computeNaturalGrowth(lvalues[i], lloss_value)
                tgvals = np.sum([bgvals, lgvals], axis=0)

            bgvals = np.subtract(bgvals, bval_lost)
            lgvals = np.subtract(lgvals, lval_lost)
            tgvals = np.subtract(tgvals, tval_lost)

            bvalues.append(bgvals)
            lvalues.append(lgvals)
            tvalues.append(tgvals)

        for i in range(len(bvalues)):
            b_field_name = "Build_Value_Year_" + str(i)
            l_field_name = "Land_Value_Year_" + str(i)
            t_field_name = "Total_Value_Year_" + str(i)

            self.file.add_column(b_field_name, bvalues[i])
            self.file.add_column(l_field_name, lvalues[i])
            self.file.add_column(t_field_name, tvalues[i])

        self.file.save_csv('flood_model.csv')
        print("Simulation complete...")
        return bvalues, lvalues, tvalues


    # Note this method does not compute complete results do to value overloading
    def summarize_total_impact(self, start=0, stop=10):
        summary = []
        for i in range(start, stop):
            t_field_name = "Total_Value_Year_" + str(i)
            data = self.file.get_column(t_field_name)
            # There is a bug where summing over positive values results in negatives
            median_data = statistics.median(data[1:][0])
            # summed_data = np.sum(data[1:][0], dtype=np.float64)
            summary.append(median_data)
        return summary

    # Note this method does not compute complete results do to value overloading
    def summarize_building_impact(self, start=0, stop=10):
        summary = []
        for i in range(start, stop):
            t_field_name = "Total_Build_Value_Year_" + str(i)
            data = self.file.get_column(t_field_name)
            # There is a bug where summing over positive values results in negatives
            median_data = statistics.median(data[1:][0])
            # summed_data = np.sum(data[1:][0], dtype=np.float64)
            summary.append(median_data)
        return summary

    # Note this method does not compute complete results do to value overloading
    def summarize_land_impact(self, start=0, stop=10):
        summary = []
        for i in range(start, stop):
            t_field_name = "Total_Land_Value_Year_" + str(i)
            data = self.file.get_column(t_field_name)
            # There is a bug where summing over positive values results in negatives
            median_data = statistics.median(data[1:][0])
            # summed_data = np.sum(data[1:][0], dtype=np.float64)
            summary.append(median_data)
        return summary


    def compute_impact_intensity(self, elevations, brisk_zones, bloss_fun):
        data = []
        sums = []
        means = []
        for i in range(len(brisk_zones)):
            field_name = "Impact_Intensity_" + str(i)
            impact_multiplier = self.computeLossFunctionZ(elevations, bloss_fun, i, brisk_zones[i])
            sums.append(np.sum(impact_multiplier, dtype=np.int64))
            means.append(statistics.mean(impact_multiplier))
            data.append(impact_multiplier)
            self.file.add_column(field_name, impact_multiplier)
        return data, sums, means

    # Get all critical facilities and their zones for each flood scenario
    # The percolation of q given an elevation threshold will be used to compute a risk metric.
    def get_critical_zones(self, cf_field, threshold):
        data = []
        impact_field_name = 'Impact_Zones_' + str(threshold)
        impact_zones = self.file.get_column(impact_field_name)
        cfs = self.file.get_column(cf_field)
        for i in range(len(cfs[1:][0])):
            if len(cfs[1:][0][i]) > 0:
                data.append((impact_zones[1:][0][i], cfs[1:][0][i]))
        return data


    def get_critical_facilities(self, cf_field):
        data = []
        cfs = self.file.get_column(cf_field)
        for i in range(len(cfs[1:][0])):
            if len(cfs[1:][0][i]) > 0:
                data.append(self.file.output[1:][i])
        return data


    def event_randomizer(self, min_intensity, max_intensity, multiplier):
        pool = []
        for i in range(max_intensity):
            v = len(range(max_intensity*multiplier)) - (i*multiplier)
            row = [i] * v
            pool.extend(row)
        maxm = len(pool) - 1
        num = random.randint(min_intensity, maxm)
        return pool[num]



