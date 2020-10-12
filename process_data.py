#!/usr/bin/env python
# coding: utf-8

# Import dependencies and pollen data. 260771 total entries!

import h5py
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import time

f = h5py.File('pollen_scraped.h5', 'r')
proc_data = dict()

# Names of the pollen measuring stations and types of pollen:

excluded = ['guadalajara', 'pontevedra', 'santiago-compostela']

pollen_names = ['alnus', 'alternaria', 'artemisa', 'betula', 'carex',
                'castanea', 'cupresaceas', 'fraxinus', 'gramineas', 'mercurialis',
                'morus', 'olea', 'palmaceas', 'pinus', 'plantago', 'platanus',
                'populus', 'amarantaceas', 'quercus', 'rumex', 'ulmus', 'urticaceas']

pollen_stations = ['albacete', 'alcazar', 'alicante', 'almeria', 'avila', 'badajoz',
         'barcelona', 'barcelona-uab', 'bejar', 'bilbao', 'burgos', 'burjassot', 'caceres',
         'cadiz', 'cartagena', 'castellon-de-la-plana', 'ciudad-real', 'cordoba', 'coruña',
         'cuenca', 'elche', 'gerona', 'granada', 'gijon', 'guadalajara', 'huelva', 'huesca',
         'jaen-hospital', 'jaen', 'jativa', 'las-palmas', 'leon', 'lerida', 'logroño',
          'madrid-subiza', 'madrid-hospital', 'malaga', 'murcia', 'oviedo', 'palencia',
          'palma-mallorca', 'pamplona', 'ponferrada', 'pontevedra', 'salamanca', 'san-sebastian',
          'santa-cruz-tenerife', 'santander', 'santiago-compostela', 'segovia', 'sevilla-macarena',
          'sevilla-tomillar', 'soria', 'talavera', 'tarragona', 'teruel', 'toledo', 'torrelavega',
           'tudela', 'valencia', 'valladolid', 'vitoria', 'zamora', 'zaragoza']

weather_stations = {'albacete': '8178D', 'alcazar':'4121', 'alicante': '8025', 'almeria': '6325O',
                    'avila': '2444', 'badajoz' : '4452', 'barcelona': '0076', 'barcelona-uab': '0200E',
                    'bejar': '2870', 'bilbao': '1082', 'burgos':'2331', 'burjassot':'8414A', 
                    'caceres': '3469A', 'cadiz': '5973', 'cartagena': '7012C', 'castellon-de-la-plana':'8500A',
                   'ciudad-real': '4121', 'cordoba': '5402', 'coruña': '1387', 'cuenca': '8096',
                   'elche': '8019', 'gerona': '0367', 'granada':'5530E', 'gijon': '1208H', 'guadalajara': '3168C',
                   'huelva': '4642E', 'huesca': '9898', 'jaen-hospital': '5270B', 'jaen': '5270B',
                   'jativa': '8293X', 'las-palmas': 'C029O', 'leon': '2661', 'lerida': '9771C', 'logroño':'9170',
                   'madrid-subiza': '3196', 'madrid-hospital': '3194U', 'malaga': '6155A', 'murcia': '7012C',
                   'oviedo': '1249I', 'palencia': '2400E', 'palma-mallorca': 'B278', 'pamplona': '9262',
                   'ponferrada': '1549', 'pontevedra': '1484C', 'salamanca': '2870', 'san-sebastian': '1024E',
                   'santa-cruz-tenerife': 'C449C', 'santander': '1111', 'santiago-compostela': '1475X', 
                   'segovia': '2465', 'sevilla-macarena': '5783', 'sevilla-tomillar': '5783', 'soria':'2030',
                   'talavera': '3365A', 'tarragona': '0016A', 'teruel': '8368U', 'toledo': '3260B', 
                   'torrelavega': '1109', 'tudela': '9434', 'valencia': '8416', 'valladolid': '2422',
                   'vitoria': '9091O', 'zamora': '2614', 'zaragoza': '9434'}


# We will write all the data processing steps in functions to apply them to each pollen station later

# We retrieve the pollen data(only cupresaceae at the moment) and eliminate outliers(values > 1000 or < 0)

def get_pollen_data(station):
    pollen_data = np.array(f[station][:, [0,7]])
    pollen_data[:, 1] = np.maximum(np.minimum(pollen_data[:, 1], 1000), 0)
    
    print('pollen data of', station, 'is of shape', pollen_data.shape)
    
    return pollen_data


# The first column of the pollen data is the date of test, in format YYYYMMDD type=int32, so we will use this method to convert it to python's date format(datetime)

def integer_to_date(integer):
    day = integer % 100
    integer = int((integer - day)/100)
    month = integer % 100
    year = int((integer - month)/100)
    
    return date(year, month, day)


# This function computes the start and end date of the pollen data, and prints it

def get_date_range(pollen_data):
    start_date = integer_to_date(pollen_data[0, 0])
    end_date = integer_to_date(pollen_data[-1, 0])

    print('Tenemos datos desde el {} hasta el {}'.format(start_date, end_date))
    
    return start_date, end_date


# REST API call to get weather data from AEMET. So easy compared to the pollen data lol

def readAEMETData(url):
    response = requests.request("GET", url)
    
    try:
        return response.json()
    except ValueError as e:
        print('429 recogiendo datos ._.')
        
        time.sleep(5)
        return readAEMETData(url)


def getAEMETbatch(fechaini, fechafin, estacion):
    url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{}/fechafin/{}/estacion/{}".format(fechaini, fechafin, estacion)
    querystring = {"api_key":"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJib2xpdG8yaGRAZ21haWwuY29tIiwianRpIjoiMjUyOTY3MDQtZDYzZS00Zjk2LWE3NTktYTY4MzE2NzVmMjE0IiwiaXNzIjoiQUVNRVQiLCJpYXQiOjE2MDE2NTY1NzQsInVzZXJJZCI6IjI1Mjk2NzA0LWQ2M2UtNGY5Ni1hNzU5LWE2ODMxNjc1ZjIxNCIsInJvbGUiOiIifQ.KzMsubyf4Ux1jxAgu5cGKGZ7rUaGYUreYu8AR0isWjM"}

    headers = {
        'cache-control': "no-cache"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    print('weather_search:', response.json()['estado'])
    #print(response.text)
    if response.json()['estado'] == 429:
        time.sleep(5)
        return getAEMETbatch(fechaini, fechafin, estacion)
    
    return readAEMETData(response.json()['datos'])


# Taking into account the maximum possible data in one query, 5 years, we have to make multiple requests to the AEMET API

def getAEMETdata(start_date, end_date, station):

    loop_date = start_date - relativedelta(days=1)
    weather_data = []

    while (end_date - loop_date).days >= 0:

        loop_date = loop_date + relativedelta(days=1)
        fechaini = datetime.strftime(loop_date, '%Y-%m-%d') + 'T00:00:00UTC'

        loop_date = loop_date + relativedelta(years = 5, days=-7)

        if (end_date - loop_date).days < 0:
            fechafin = datetime.strftime(end_date, '%Y-%m-%d') + 'T23:59:59UTC'
        else:
            fechafin = datetime.strftime(loop_date, '%Y-%m-%d') + 'T23:59:59UTC'

        #print('fechaini', fechaini, 'fechafin', fechafin)
        weather_data = weather_data + getAEMETbatch(fechaini, fechafin, weather_stations[station])
    
    return weather_data


# The pollen data has multiple holes, so we eliminate them from the weather data too, to make the dates match. The below code takes care of it, i'm quite proud of it uwu

# In some stations the weather data has holes too!! xddd
# We return the new pollen_data because numpy delete creates a copy of the original array while this isn't needed for weather_data

def delete_holes(pollen_data, weather_data):
    i = 1
    pollen_holes = 0
    weather_holes = 0
    
    while i < min(pollen_data.shape[0], len(weather_data)):
        pollen_date = integer_to_date(pollen_data[-i, 0]) 
        weather_date = datetime.strptime(weather_data[-i]['fecha'], '%Y-%m-%d').date()
        
        date_difference = (pollen_date - weather_date).days
        
        if date_difference > 0:
            pollen_data = np.delete(pollen_data, pollen_data.shape[0] - i, axis=0)
            weather_holes += 1
        
        if date_difference < 0:
            del weather_data[-i]
            pollen_holes += 1

        if date_difference == 0:
            i += 1             
    
    print('there were {} pollen holes and {} weather holes'.format(pollen_holes, weather_holes))
    
    pollen_data = pollen_data[-min(pollen_data.shape[0], len(weather_data)):]
    weather_data = weather_data[-min(pollen_data.shape[0], len(weather_data)):]
        
    return pollen_data


# We have to check that the pollen and weather data have the same size and that their dates match up. We then return the training size, $m$

def verify_integrity(pollen_data, weather_data):
    print('m=', len(weather_data))
    
    assert(pollen_data.shape[0] == len(weather_data))
    
    m = len(weather_data)
    
    for i in range(m):
        if integer_to_date(pollen_data[-(i + 1), 0]) != datetime.strptime(weather_data[-(i + 1)]['fecha'], '%Y-%m-%d').date():
            print('error at {}, pollen_date is {} and weather_date is {}'.format(i , integer_to_date(pollen_data[-(i + 1), 0]), datetime.strptime(weather_data[-(i + 1)]['fecha'], '%Y-%m-%d').date()))
        assert(integer_to_date(pollen_data[-(i + 1), 0]) == datetime.strptime(weather_data[-(i + 1)]['fecha'], '%Y-%m-%d').date())
        
    print('YAHOO!')


# We transform the day of the year to a sinusoidal wave, to account for season differences in a periodic way(0 = 365)

def get_year_position(pollen_dates):
    m = pollen_dates.shape[0]
    
    year_position = np.zeros((m, 2))
    
    for i in range(m):
        day_of_year = float(integer_to_date(pollen_dates[i]).strftime('%j'))
        year_position[i][0] = math.cos(day_of_year*2*math.pi/365)
        year_position[i][1] = math.sin(day_of_year*2*math.pi/365)
    return year_position


# Lets format the data into a np.array to feed the RNN
# 
# n is the number of features, in our case:
# 
# - Sin-wave of the day of the year
# - Cos-wave of the day of the year
# - Pollen level
#     - Here I compute the final data as $z = log(x + 1)$. I pass it through this log kernel because I suspect that it is what I will use when I classify the predictions into a few classes or 'levels' of pollen in air. This is because, a jump in pollen levels of a fixed amount is much more noticeable if it comes from a low value(where the user might jump from no symptoms to light symptoms) that in a already high value where the user will be fucked up either way.
# - Max temperature
# - Mean temperature
# - Min temprerature
# - Max pressure
# - Min pressure
# - Mean wind speed
# - Max wind speed
# - Precipitations
# - Wind component in each direction(we compute this separatedly from the other params)
#     - I noticed that the parameter of direction is in TENS of degrees, so we have to multiply it by ten. Oh, and I forgot to change it to radians so it was basically useless xd
#     
# Guess what, the weather data also has holes! And in each category separately!
# To combat this we will compute exponentially weighted means to use when some parameter is not known

params = ['time_sin','time_cos', 'pollen', 'dx', 'dy', 'tmax', 'tmed', 'tmin', 'presMax', 'presMin', 'velmedia', 'racha', 'prec', 'altitud']
n = len(params)

print('we have {} parameters'.format(n))

def process_data(pollen_data, weather_data):
    
    m = len(weather_data)
    
    proc_data = np.zeros((m, n), dtype=np.float32)

    proc_data[:, :2] = get_year_position(pollen_data[:, 0])
    proc_data[:, 2] = np.log(pollen_data[:, 1] + 1)

    beta = 0.9
    exp_means = np.zeros(n)
    holes = np.zeros(n, dtype=np.int32)

    straight_data_index = 5
    
    for i in range(m):
        if 'prec' in weather_data[i]:
            #print(weather_data[i]['prec'])
            if weather_data[i]['prec'] == 'Ip':
                weather_data[i]['prec'] = '0,0'
        if 'dir' in weather_data[i]:
            angle = float(weather_data[i]['dir'].replace(',', '.'))*math.pi/18

            proc_data[i, straight_data_index - 2] = math.cos(angle)
            proc_data[i, straight_data_index - 1] = math.sin(angle)

            exp_means[straight_data_index - 2] = beta*exp_means[straight_data_index - 2] + (1 - beta)*math.cos(angle)
            exp_means[straight_data_index - 1] = beta*exp_means[straight_data_index - 1] + (1 - beta)*math.sin(angle)
        else:
            proc_data[i, straight_data_index - 2] = exp_means[straight_data_index - 2]/(1-beta**(i + 1))
            proc_data[i, straight_data_index - 1] = exp_means[straight_data_index - 1]/(1-beta**(i + 1))

            holes[straight_data_index - 2] += 1
            holes[straight_data_index - 1] += 1

        #We start at 3 because we compute wind direction components separately
        for j in range(straight_data_index, n):       
            if params[j] in weather_data[i]:
                try:
                    proc_data[i, j] = float(weather_data[i][params[j]].replace(',', '.'))
                    exp_means[j] = beta*exp_means[j] + (1 - beta)*proc_data[i, j]
                except:
                    print('exception')
                    proc_data[i, j] = exp_means[j]/(1-beta**(i + 1))
                    holes[j] += 1
            else:
                proc_data[i, j] = exp_means[j]/(1-beta**(i + 1))
                holes[j] += 1
                
        for j in range(n):
            if np.isnan(proc_data[i, j]):
                print(i, params[j])
                print('NaN')
                proc_data[i, j] = 0

    print('holes in each parameter:', holes)
    print('-----------------------------------')
    return proc_data


# Alright, now we should normalize each parameter. We also store the mean and standard deviation of the pollen distribution to rectify the predictions later. We have to take care to normalize ALL the pollen data in a single distribution, independently of the station We also exclude the altitude from this

# Now we will split the data in windows of analysis_size + prediction_size, where the RNN will analyze the first analysis_size days to predict pollen levels in the last prediction_size days


def sliding_windows(data, window_size):
    windows = np.zeros((data.shape[0] - window_size + 1, window_size, data.shape[1]))
    
    for i in range(data.shape[0] - window_size + 1):
        windows[i] = data[i:i+window_size]
        
    return windows

def build_proc_data():
    # No tenemos buenos datos meteorologicos para guadalajara ni pontevedra. Hay un problema con el polen de santiago de compostela

    start = 'albacete'
    startIndex = pollen_stations.index(start)

    for station in pollen_stations[startIndex:]:
        if not station in excluded:
            pollen_data = get_pollen_data(station)
            start_date, end_date = get_date_range(pollen_data)

            weather_data = getAEMETdata(start_date, end_date, station)

            pollen_data = delete_holes(pollen_data, weather_data)
            verify_integrity(pollen_data, weather_data)

            proc_data[station] = process_data(pollen_data, weather_data)

def save_proc_data():
    proc_file = h5py.File('proc_data.h5', 'a')

    # Write proc_data to the file, before normalizing and splitting

    for station in pollen_stations:
        if not station in excluded:
            try:
                del proc_file[station]
            except:
                print('No existe database para', station)
            proc_file.create_dataset(station, data=proc_data[station])
    proc_file.close()


# Normalize data
def normalize_data(proc_data):
    
    n = proc_data['barcelona'].shape[1]
    
    mean = np.zeros(n - 1)
    var = np.zeros(n - 1)
    data_count = 0
    
    for station in proc_data.keys():
        local_count = proc_data[station].shape[0]
        data_count += local_count
        
        for j in range(n - 1):
            mean[j] += proc_data[station][:, j].mean()*local_count
            #in-group variance
            var[j] += proc_data[station][:, j].var()*local_count
            
    
    mean /= data_count
    
    for station in proc_data.keys():
        local_count = proc_data[station].shape[0]
        
        #Outside-variance
        for j in range(n - 1):
            var[j] += local_count*(proc_data[station][:, j].mean() - mean[j])**2
    
    var /= data_count
    std = np.sqrt(var)
    
    for station in proc_data.keys():
        for j in range(n - 1):
            proc_data[station][:, j] = (proc_data[station][:, j] - mean[j])/std[j]
            
        #print(math.sqrt(proc_data[station][:, 0].var()))
          
    parameters = np.zeros((2, n - 1))
    parameters[0] = mean
    parameters[1] = std
    return parameters


def read_proc_data():
    proc_file = h5py.File('proc_data.h5', 'a')

    for station in pollen_stations:
        if not station in excluded:
            proc_data[station] = proc_file[station]

    proc_file.close()


# Save mean and std of all data for dev/test
def save_parameters(parameters):
    proc_file = h5py.File('proc_data.h5', 'a')

    proc_file.create_dataset('parameters', data=parameters)
    proc_file['parameters'][...] = parameters

    proc_file.close()


def split_data(XY_total, analysis_size, prediction_size, train_rate, dev_rate):
    window_size = analysis_size + prediction_size

    np.random.shuffle(XY_total)

    train_set = XY_total[:int(XY_total.shape[0] * train_rate), :, :]
    dev_set = XY_total[int(XY_total.shape[0] * train_rate):int(XY_total.shape[0] * (train_rate + dev_rate)), :, :]
    test_set = XY_total[int(XY_total.shape[0] * (train_rate + dev_rate)):, :, :]

    X_train = train_set
    Y_train = train_set[:, analysis_size:window_size, 2]

    X_dev = dev_set
    Y_dev = dev_set[:, analysis_size:window_size, 2]

    X_test = test_set
    Y_test = test_set[:, analysis_size:window_size, 2]

    print('X_train.shape', X_train.shape)
    print('X_dev.shape', X_dev.shape)
    print('X_test.shape', X_test.shape, '\n')

    print('Y_train.shape', Y_train.shape)
    print('Y_dev.shape', Y_dev.shape)
    print('Y_test.shape', Y_test.shape)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test


def save_train_data(anal_size, pred_size, train_rate=0.85, dev_rate=0.1):
    proc_file = h5py.File('proc_data.h5', 'a')

    window_size = anal_size + pred_size
    XY_total = np.zeros((0, window_size, n))

    for station in proc_data.keys():
        XY_total = np.append(XY_total, sliding_windows(proc_data[station], window_size), axis=0)

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_data(XY_total, anal_size, window_size, train_rate, dev_rate)

    try:
        del proc_file['X_train']
        del proc_file['Y_train']
        del proc_file['X_dev']
        del proc_file['Y_dev']
        del proc_file['X_test']
        del proc_file['Y_test']
    except:
        print('falta alguna database')

    proc_file.create_dataset('X_train', data=X_train)
    proc_file.create_dataset('Y_train', data=Y_train)
    proc_file.create_dataset('X_dev', data=X_dev)
    proc_file.create_dataset('Y_dev', data=Y_dev)
    proc_file.create_dataset('X_test', data=X_test)
    proc_file.create_dataset('Y_test', data=Y_test)

    proc_file.close()
