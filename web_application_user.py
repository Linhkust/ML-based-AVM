import math
import os
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect
import pymysql
import json
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
import shap
from joblib import dump, load

'''
Estate Information Panel
'''
def estate_information(district, name):
    # Estate Information
    estate = pd.read_csv('..\\ML based AVM\\data\\Estate.csv')
    selected_estate = estate[(estate['Estate'] == name) & (estate['Region'] == district)]

    # Get the information
    address = selected_estate.loc[:, 'Address'].reset_index(drop=True).iloc[0]
    min_age = selected_estate.loc[:, 'MinBuildingAge'].reset_index(drop=True).iloc[0]
    max_age = selected_estate.loc[:, 'MaxBuildingAge'].reset_index(drop=True).iloc[0]
    age = str(min_age) +'-' + str(max_age)
    phase_no = selected_estate.loc[:, 'PhaseNum'].reset_index(drop=True).iloc[0]
    block_no = selected_estate.loc[:, 'BuildingNum'].reset_index(drop=True).iloc[0]
    unit_no = selected_estate.loc[:, 'UnitNum'].reset_index(drop=True).iloc[0]
    developer = selected_estate.loc[:, 'Developer'].reset_index(drop=True).iloc[0]
    school = selected_estate.loc[:, 'SchoolNetwork'].reset_index(drop=True).iloc[0]

    dict = {'Estate': name,
            'Address': address,
            'Age': int(min_age),
            'Age_Range': age,
            'Phase_No': phase_no,
            'Block_No': block_no,
            'Unit_No': unit_no,
            'Developer': developer,
            'SchoolNet': school
    }

    return dict

"""
POI Information Panel
"""


# Hong Kong Geo Store Geocoding API
def geocoding(address):
    url = 'https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q=' + address
    response = requests.get(url).text
    r = json.loads(response)
    east = r[0]['x']
    north = r[0]['y']
    return east, north


# WIFI_HK information
def wifi_hk(east, north):
    wifi = pd.read_csv('..\\ML based AVM\\data\\WIFI_HK.csv')
    selected_wifi = wifi.loc[(east - 1000 <= wifi['Easting']) &
                           (wifi['Easting'] <= east + 1000) &
                           (north - 1000 <=wifi['Northing']) &
                           (wifi['Northing'] <= north + 1000)]

    wifi = selected_wifi.copy().reset_index()

    temp = []
    for i in range(len(wifi)):
        distance = math.sqrt((east - wifi.loc[i, 'Easting']) ** 2 + (north - wifi.loc[i, 'Northing']) ** 2)
        if distance > 500:
            temp.append(i)
        else:
            continue

    wifi.drop(index=temp, inplace=True)
    wifi = wifi.reset_index()
    return len(wifi)

# Suitable POI information
def poi_information(east, north):
    poi = pd.read_excel('..\\ML based AVM\\data\\GeoCom.xlsx', engine="openpyxl")
    selected_poi = poi.loc[(east-1000 <= poi['EASTING']) &
                           (poi['EASTING'] <= east+1000) &
                           (north-1000 <= poi['NORTHING']) &
                            (poi['NORTHING'] <= north+1000)]
    poi = selected_poi.copy().reset_index()
    temp = []
    for i in range(len(poi)):
        distance = math.sqrt((east - poi.loc[i, 'EASTING']) ** 2 + (north - poi.loc[i, 'NORTHING']) ** 2)
        if distance > 500:
            temp.append(i)
        else:
            continue
    poi.drop(index=temp, inplace=True)

    poi_class = len(poi.loc[:, 'CLASS'].unique())
    poi_types = len(poi.loc[:, 'TYPE'].unique())

    # Select the poi type
    variable=["SMK", "RGD",	"BUS", "MIN", "MAL",
              "CVS", "RCP", "CPO", "SGD", "PLG",
              "BAS", "CMC", "TOI", "POB", "KDG", "CHU",
              "PRS", "SES", "EES", "PAV", "LIB"]

    # POI Diversity
    poi_num = []
    poi_dict = {}
    for i, poi_type in enumerate(variable):
        num = len(poi[poi['TYPE'] == poi_type])
        poi_num.append(num)
        poi_dict[poi_type] = num

    poi_num = np.array(poi_num)
    poi_num = poi_num / np.sum(poi_num)
    poi_diversity = []
    for i, poi_value in enumerate(poi_num):
        poi_diversity.append(-poi_value * math.log2(poi_value) if poi_value > 0 else 0)

    poi_diversity = format(sum(poi_diversity), '.3f')
    wifi = {'WIFI_HK': wifi_hk(east, north)}

    poi_dict = {**poi_dict, **wifi}

    poi_info = {'POI_Class': poi_class,
                'POI_Type': poi_types,
                'POI_Diversity': float(poi_diversity)
                }
    poi_info = {**poi_info, **poi_dict}

    return poi_info

"""
CCL Index Panel
"""
def ccl_index(address):
    index = pd.read_csv('..\\ML based AVM\\data\\CCL.csv')

    NTE = ['SHA TIN DISTRICT',
           'TAI PO DISTRICT',
           'NORTH DISTRICT',
           'SAI KUNG DISTRICT'
           ]

    NTW = ['YUEN LONG DISTRICT',
           'TUEN MUN DISTRICT',
           'TSUEN WAN DISTRICT',
           'KWAI TSING DISTRICT',
           'ISLANDS DISTRICT'
    ]

    # Return the district info
    headers={'Accept':'application/json',
             'Accept-Language':'en'
    }
    url = 'https://www.als.ogcio.gov.hk/lookup?q=' + address
    response = requests.get(url, headers=headers).text
    r = json.loads(response)

    district = r['SuggestedAddress'][0]['Address']['PremisesAddress']['EngPremisesAddress']['EngDistrict']['DcDistrict']
    region = r['SuggestedAddress'][0]['Address']['PremisesAddress']['EngPremisesAddress']['Region']

    # Choose the corresponding ccl index
    if region == 'NT':
        if district in NTE:
            ccl = index.loc[0, 'NTE']
        else:
            ccl = index.loc[0, 'NTW']
    elif region == 'HK':
        ccl = index.loc[0, 'HK']
    else:
        ccl = index.loc[0, 'KL']

    return {'CCL': float(ccl)}


"""
Property Valuation Model
"""
def RF_model():
    # Real Time Update

    # Update the database
    data = pd.read_csv('Final_data.csv')

    # update the features and parameters

    # Update the feature results
    refcv_results = [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 9,
                     12, 10, 4, 8, 11, 5, 6, 7, 9, 7, 5, 12, 13, 3, 10, 6, 8,
                     11, 13, 4]

    selected_features = [x for x, y in enumerate(refcv_results) if y == 1]
    housing_x = data.iloc[:, selected_features].values
    housing_y = data.iloc[:, -1].values.ravel()

    # Model train
    # Update the model parameter
    model = RandomForestRegressor(n_estimators=128, max_depth=16)
    model.fit(housing_x, housing_y)
    # dump(model, '..\\ML based AVM\\data\\RF.joblib')


def valuation(input_x):
    model = load('..\\ML based AVM\\data\\RF.joblib')

    final_valuation = model.predict(np.array(input_x).reshape(1, -1))[0]

    explainer = shap.explainers.Tree(model)

    # shap results
    values = explainer(np.array(input_x))
    print(values)

    base_valuation = format(values.base_values[0], '.2f') + 'M'
    price_premium = format(np.sum(values.values), '.2f') + 'M'
    final_valuation = format(final_valuation, '.2f') + 'M'

    return base_valuation, price_premium, final_valuation


'''
Web Interface
'''

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('user.html')

@app.route('/User/result', methods=['POST', 'GET'])
def user():
    if request.method == "POST":
        #
        district= request.form.get('district')
        estate = request.form.get('estate')
        block = request.form.get('block_tower')
        floor_level = int(request.form.get('floor_level'))
        gfa = float(request.form.get('gfa'))

        # Estate Information
        content_EI = estate_information(district, estate)

        address = estate + ' ' + block
        east = geocoding(address)[0]
        north = geocoding(address)[1]

        # POI Information
        content_POI = poi_information(east, north)

        # CCL Index
        content_CCL = ccl_index(address)

        content = {**content_EI, **content_POI, **content_CCL}

        # Input_X
        ccl = content_CCL['CCL']
        age =content_EI['Age']
        building_num = content_EI['Block_No']
        unit_num = content_EI['Unit_No']
        del content_POI['POI_Class']
        del content_POI['CMC']
        del content_POI['LIB']
        input_x = [ccl, int(floor_level), age, building_num, unit_num, east, north, gfa] + list(content_POI.values())

        result = valuation(input_x)

        valuation_result = {
        'Base_Valuation': result[0],
        'Price_Premium': result[1],
        'Prediction_Result': result[2]
        }

        content ={**content, **valuation_result}

        return render_template('user.html', **content)


if __name__ == "__main__":
    # print(valuation_model(input_x))
    # estate_information('HK', 'Kornhill')
    app.run(debug=True)

