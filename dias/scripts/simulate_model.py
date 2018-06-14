import random
from dias.scripts.base_model import *


def simulate_base_model(model, building_val_field, land_val_field, intensity_range, iterations=10, time_step=10,
                       output="file.csv"):
    """

    :param model: processdbf object | contains
    :param building_val_field: 
    :param land_val_field: 
    :param intensity_range: 
    :param iterations: 
    :param time_step: 
    :param output: 
    :return: 
    """
    try:
        print("Initializing simulation...")
        bvalue = [np.array(model[4].get_column(building_val_field)[1], dtype='float64')]
        lvalue = [np.array(model[4].get_column(land_val_field)[1], dtype='float64')]

        bvalue_change = [np.zeros(len(bvalue[0]))]
        lvalue_change = [np.zeros(len(lvalue[0]))]

        for i in range(time_step):
            print("Iteration... ", str(i + 1))
            abv = []
            alv = []
            for j in range(iterations):
                e = event_randomizer(intensity_range[0], intensity_range[1], 1.0)

                impact = np.array(model[3][0][e], dtype='float64')
                bloss = computeLoss(bvalue[i], model[0][e], impact, 0.075)
                bpattern = invert_pattern(model[0][e])
                bgrowth = computeGrowth(bvalue[i], bpattern, 0.02)

                land_model = np.divide(impact, 2)
                lloss = computeLoss(lvalue[i], model[0][e], land_model, 0.03)
                lpattern = invert_pattern(model[0][e])
                lgrowth = computeGrowth(lvalue[i], lpattern, 0.001)

                bv = bvalue[i] + (bgrowth - bloss)
                lv = lvalue[i] + (lgrowth - lloss)

                abv.append(bv)
                alv.append(lv)

            newb = np.array(abv).mean(axis=0)
            btest = np.divide(newb, bvalue[0])

            bval_title = 'Building_Value_Year_' + str(i)
            model[4].add_column(bval_title, newb)
            bvalue.append(newb)

            perc_change = "BValue_Change_Year_" + str(i)
            nans = np.isnan(btest)
            btest[nans] = 0
            model[4].add_column(perc_change, btest)
            bvalue_change.append(btest)

            newl = np.array(alv).mean(axis=0)
            ltest = np.divide(newl, lvalue[0])

            lval_title = 'Land__Value_Year_' + str(i)
            model[4].add_column(lval_title, newl)
            lvalue.append(newl)

            perc_change = "LValue_Change_Year_" + str(i)
            nans = np.isnan(ltest)
            ltest[nans] = 0
            model[4].add_column(perc_change, ltest)
            lvalue_change.append(ltest)

        model[4].save_csv(output)
        return bvalue_change, lvalue_change, model[4]
    except ValueError:
        pass
    except RuntimeError:
        pass


def impact_by_zone(model, parcel_field, impact_zone_field):
    """
    :param model: object reference to data
    :param parcel_field: string
    :param impact_zone_field: string
    :return: 
    """
    try:
        zones = model[2].get_column(impact_zone_field)[1]
        bvalue = np.array(model[0]).mean(axis=0)
        lvalue = np.array(model[1]).mean(axis=0)
        bdata = []
        ldata = []
        for i in range(int(max(zones) + 1)):
            btmp = []
            ltmp = []
            print("Getting zones ", str(i))
            for j in range(len(zones)):
                if int(zones[j]) == i:
                    btmp.append(bvalue[j])
                    ltmp.append(lvalue[j])
                else:
                    pass
            bdata.append(np.median(btmp))
            ldata.append(np.median(ltmp))

        baverages = []
        laverages = []
        for i in range(len(bdata) + 1):
            for j in range(len(zones)):
                if int(zones[j]) == i:
                    baverages.append(bdata[i])
                    laverages.append(ldata[i])
        bfield = "Change_Ratio_B"
        lfield = "Change_Ratio_L"
        print(len(baverages), len(zones), len(bvalue))
        model[2].add_column(bfield, baverages)
        model[2].add_column(lfield, laverages)
        print("Saved - Simulated Value Change by Zone")
        return model[2]
    except ValueError:
        pass
    except RuntimeError:
        pass


def event_randomizer(min_intensity, max_intensity, multiplier):
    """
    :param min_intensity: 
    :param max_intensity: 
    :param multiplier: 
    :return: 
    """
    pool = []
    for i in range(max_intensity):
        v = len(range(int(max_intensity))) - (i * multiplier)
        row = [i] * int(v)
        pool.extend(row)
    maxm = len(pool) - 1
    num = random.randint(min_intensity, maxm)
    return pool[num]


def computeLoss(values, pattern, zone_vector, loss_const):
    """
    :param values: 
    :param impact_multiplier: 
    :param zone_vector: 
    :return: 
    """

    npattern = computePattern(zone_vector, pattern)
    new = loss_const * npattern
    loss = computePattern(values, new)
    # new_val = values - loss
    return loss


def computeGrowth(values, pattern, growth_rate):
    """
    :param values: 
    :param pattern: 
    :param growth_rate: 
    :return: 
    """
    vpattern = computePattern(values, pattern)
    growth = vpattern * growth_rate
    return growth


# Get all critical facilities and their zones for each flood scenario
# The percolation of q given an elevation threshold will be used to compute a risk metric.
def get_critical_zones(cf_field, threshold, filename):
    """
    :param cf_field: 
    :param threshold: 
    :param filename: 
    :return: 
    """
    file = processdbf(filename)
    file.open_csv()
    data = []
    impact_field_name = 'Impact_Zones_' + str(threshold)
    impact_zones = file.get_column(impact_field_name)
    cfs = file.get_column(cf_field)
    for i in range(len(cfs[1:][0])):
        if len(cfs[1:][0][i]) > 0:
            data.append((impact_zones[1:][0][i], cfs[1:][0][i]))
    return data


def get_critical_facilities(cf_field, filename):
    """
    :param cf_field: 
    :param filename: 
    :return: 
    """
    file = processdbf(filename)
    file.open_csv()
    data = []
    cfs = file.get_column(cf_field)
    for i in range(len(cfs[1:][0])):
        if len(cfs[1:][0][i]) > 0:
            data.append(file.output[1:][i])
    return data


def run_model(file, file2, lat, lon,  building_value_field, land_value_field, parcel_field, impact_field,
              impact_range, max_impact, impact_multiplier, iterations, time_step,
                             output_file_name):
    """
    :param file: 
    :param file2: 
    :param lat: 
    :param lon: 
    :param building_value_field: 
    :param land_value_field: 
    :param parcel_field: 
    :param impact_field: 
    :param impact_range: 
    :param max_impact: 
    :param impact_multiplier: 
    :param iterations: 
    :param time_step: 
    :param output_file_name: 
    :return: 
    """
    model = build_base_model(file, file2, lat, lon, max_impact, impact_multiplier)
    sim = simulate_base_model(model, building_value_field, land_value_field, impact_range, iterations, time_step,
                             output_file_name)
    zones = impact_by_zone(sim, parcel_field, impact_field)
    return zones
