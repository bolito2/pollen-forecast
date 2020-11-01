#!/usr/bin/env python
# coding: utf-8

# Import dependencies
import os

import h5py
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import numpy as np
import math
import time

# Metadata for getting AEMET data and other things
import metadata


# Class for reading and processing the data
class DataHandler:
    # Create the dictionary for the pooled weather-pollen data and later
    # normalized data. We also initialize the mean and std arrays to None
    def __init__(self):
        self.pooled_data = dict()
        self.norm_data = dict()

        self.coordinates = None

        self.mean = None
        self.std = None
        self.max_altitude = -1

    # <<< STATIC UTILITY METHODS >>>

    # Getter of the pollen data of a given station
    @staticmethod
    def get_pollen_data(station):
        with h5py.File('pollen_scraped.h5', 'r') as pollen_scraped_file:
            pollen_data = np.array(pollen_scraped_file[station][:, [0, 7]])  # 0 for the date and 7 for cupresaceae,
            # the pollen in which we will focus now
            pollen_data[:, 1] = np.maximum(np.minimum(pollen_data[:, 1], 1000), 0)  # Clean up some outliers

            print('pollen data of', station, 'is of shape', pollen_data.shape)  # The shape is (m, 2)

        return pollen_data

    # Transform the pollen data dates, formatted as integers(YYYYMMDD) to python's date format(datetime)
    @staticmethod
    def integer_to_date(integer):
        day = integer % 100
        integer = int((integer - day) / 100)
        month = integer % 100
        year = int((integer - month) / 100)

        return date(year, month, day)

    # Get the range of dates in the pollen data
    @staticmethod
    def get_date_range(pollen_data):
        start_date = DataHandler.integer_to_date(pollen_data[0, 0])
        end_date = DataHandler.integer_to_date(pollen_data[-1, 0])

        print('Data is from {} to {}'.format(start_date, end_date))

        return start_date, end_date

    # Extracting the data from AEMET server's response
    @staticmethod
    def read_aemet_response(url):
        response = requests.request("GET", url)

        # If the server throws a 429 because it is busy, try again in 5 seconds
        try:
            return response.json()
        except ValueError as e:
            print('AEMET servers are busy, trying again...')

            time.sleep(5)
            return DataHandler.read_aemet_response(url)

    # REST API call to get read data from AEMET. So easy compared to the pollen data lol
    # It returns the weather data from station in the period fechaini-fechafin (maximum of 5 years)
    @staticmethod
    def get_aemet_batch(fechaini, fechafin, station):
        url = metadata.aemet_data_url.format(fechaini, fechafin, station)
        querystring = {"api_key": metadata.api_key}  # The API key and AEMET_url are obtained from metadata.py

        # Avoid problems with getting the same data twice
        headers = {
            'cache-control': "no-cache"
        }

        # The actual GET request and its response code
        response = requests.request("GET", url, headers=headers, params=querystring)
        print('weather_search:', response.json()['estado'])

        # If the servers are busy, wait 5 seconds and try again
        if response.json()['estado'] == 429:
            time.sleep(5)
            return DataHandler.get_aemet_batch(fechaini, fechafin, station)

        # If we get a response, return its data in JSON format
        return DataHandler.read_aemet_response(response.json()['datos'])

    # Taking into account the maximum possible data in one query, make multiple requests to the
    # AEMET API after splitting the time window into batches of 5 years
    @staticmethod
    def get_aemet_data(start_date, end_date, station):
        # Prepare loop_date variable and the weather data list
        loop_date = start_date - relativedelta(days=1)
        weather_data = []

        # While loop_date is behind end_date, set the start of the search to loop_date + 1 day(to avoid duplicating the
        # end of each window with the start of the next) and then set the end of the search to the loop_date + 5 years.
        # Substract a litte amount(1 week) to make sure it doesn't return an error
        while (end_date - loop_date).days >= 0:
            loop_date = loop_date + relativedelta(days=1)
            fechaini = datetime.strftime(loop_date, '%Y-%m-%d') + 'T00:00:00UTC'

            loop_date = loop_date + relativedelta(years=5, days=-7)

            # If loop_date just surpassed end_date, search the last interval manually
            if (end_date - loop_date).days < 0:
                fechafin = datetime.strftime(end_date, '%Y-%m-%d') + 'T23:59:59UTC'
            else:
                fechafin = datetime.strftime(loop_date, '%Y-%m-%d') + 'T23:59:59UTC'

            # Finally append the batch of data to weather_data
            weather_data = weather_data + DataHandler.get_aemet_batch(fechaini, fechafin,
                                                                      metadata.weather_stations[station])

        return weather_data

    # Get information about all stations and return a dictionary with the corresponding data
    @staticmethod
    def get_aemet_stations_info():
        url = metadata.aemet_stations_url
        querystring = {"api_key": metadata.api_key}
        coordinates = dict()  # Dict for storing the coordinates of each station

        headers = {
            'cache-control': "no-cache"
        }

        response = requests.request('GET', url, headers=headers, params=querystring)

        # If servers are busy try again in 5 seconds
        if response.json()['estado'] == 429:
            print('AEMET servers are busy...')
            time.sleep(5)

            return DataHandler.get_aemet_stations_info()
        else:
            aemet_stations = DataHandler.read_aemet_response(response.json()['datos'])  # JSON list of all the stations
            for pollen_station in metadata.pollen_stations:
                for aemet_station in aemet_stations:
                    if metadata.weather_stations[pollen_station] == aemet_station['indicativo']:
                        # Get all three coordinates in string format
                        longitude_str = aemet_station['longitud']
                        latitude_str = aemet_station['latitud']
                        altitude_str = aemet_station['altitud']

                        # We format them in here because they are the same for all data points
                        altitude = float(altitude_str)

                        # The last character of the string is the cardinal point(N/S for latitude, E/W for longitude)
                        if longitude_str[-1] == 'E':
                            longitude = float(longitude_str[:-1])
                        elif longitude_str[-1] == 'W':
                            longitude = -float(longitude_str[:-1])
                        else:
                            raise ValueError('Longitude of {} is not formatted correctly'.format(pollen_station))

                        if latitude_str[-1] == 'N':
                            latitude = float(latitude_str[:-1])
                        elif latitude_str[-1] == 'S':
                            latitude = -float(latitude_str[:-1])
                        else:
                            raise ValueError('Latitude of {} is not formatted correctly'.format(pollen_station))

                        # The coordinates are formatted without a comma and with 4 decimal places
                        latitude /= 10_000
                        longitude /= 10_000

                        # Convert them to radians
                        latitude *= np.pi / 180
                        longitude *= np.pi / 180

                        # Now input the coordinates to the dictionary. Longitude and latitude are inserted in (cos,sin) format
                        coordinates[pollen_station] = [np.cos(longitude), np.sin(longitude),
                                                       np.cos(latitude), np.sin(latitude),
                                                       altitude]
            return coordinates

    # Both the pollen and weather data have holes so this function removes them from the other dataset to make the dates
    # match. This is done by iterating both arrays backwards, and if the dates don't match, remove one element from the
    # dataset with the most recent date and repeat until they match. Then advance the index by one and keep iterating

    # Also store the location of the holes(more specifically, the index just before the hole)
    # This is to prevent data far apart to get mixed into the same window later on
    @staticmethod
    def delete_holes(pollen_data, weather_data):
        i = 1
        holes = []

        while i < min(pollen_data.shape[0], len(weather_data)):
            pollen_date = DataHandler.integer_to_date(pollen_data[-i, 0])
            weather_date = datetime.strptime(weather_data[-i]['fecha'], '%Y-%m-%d').date()

            date_difference = (pollen_date - weather_date).days

            if date_difference > 0:
                pollen_data = np.delete(pollen_data, pollen_data.shape[0] - i, axis=0)

            if date_difference < 0:
                del weather_data[-i]

            if date_difference != 0 and (len(holes) == 0 or holes[-1] != i - 1):
                holes.append(i - 1)

            if date_difference == 0:
                i += 1

        pollen_data = pollen_data[-min(pollen_data.shape[0], len(weather_data)):]
        weather_data = weather_data[-min(pollen_data.shape[0], len(weather_data)):]

        return pollen_data, weather_data, holes

    # Check that the dates in the pollen and weather data match up
    # Return a boolean telling if there is a mismatch and its index if it exists
    @staticmethod
    def verify_data(pollen_data, weather_data):
        if pollen_data.shape[0] != len(weather_data):
            return True, -1

        for i in range(len(weather_data)):
            if DataHandler.integer_to_date(pollen_data[-(i + 1), 0]) != datetime.strptime(
                    weather_data[-(i + 1)]['fecha'], '%Y-%m-%d').date():
                return True, i

        return False, -1

    # Create sinusoidal waves of the period specified returned as an array of m elements
    # TODO: Check the impact of leap years
    @staticmethod
    def get_temporal_cycle(m, period):
        cycle = np.zeros((2, m))  # Initialize the cycles

        # Compute a periodic repetition of the saw sequence {0, 1, 2, ..., period - 1}
        saw_sequence = np.mod(range(m), period)

        # Then compute the sin and cos of the sequence scaled to go from 0 to 2pi.
        # This way we get complete repeating cycles of the specified period
        cycle[0] = np.cos(saw_sequence * 2 * np.pi / period)
        cycle[1] = np.sin(saw_sequence * 2 * np.pi / period)
        return np.transpose(cycle)  # The data must be of shape (m, 2) to fit in the keras model

    # Format the data into a np.array to feed the RNN
    # n is the number of features, divided in three categories, in this order:

    # FEATURE DATA
    # The main data, consisting in pollen and weather data sampled daily.

    # Pollen level

    # Max temperature
    # Mean temperature
    # Min temprerature
    # Max pressure
    # Min pressure
    # Mean wind speed
    # Max wind speed
    # Precipitations
    # Wind component in each direction(we compute this separatedly from the other features) in a (cos, sin) fashion

    # STATION COORDINATES
    # To take into account the different conditions in each location, coordinates are included in each datapoint, with:
    # (cos, sin) of longitude
    # (cos, sin) of latitude
    # Altitude

    # TEMPORAL SPAN DATA
    # Sinusoidal waves of the temporal spans of various factors that influence pollen levels
    # According to (Nowosad et al. 2015) the most important temporal spans are 1 day, 3.5 days and more than 15 days
    # so we will use the analysis size, and then obviously the year(365)

    # Guess what, the weather data also has holes! And in each category separately!
    # To combat this we will compute exponentially weighted means to use when some parameter is not known
    @staticmethod
    def process_data(pollen_data, weather_data, coordinates):
        m = len(weather_data)  # Sample size

        # Initialize feature data
        feature_data = np.zeros((m, metadata.n_features), dtype=np.float32)

        # The first row is pollen, with a log kernel(temporal)
        feature_data[:, 0] = pollen_data[:, 1]

        # Initialize the parameters for the exp-weighted mean
        beta = 0.9
        exp_means = np.zeros(metadata.n_features)

        # This variable will hold the angle of the wind direction vector if it exists, and be equal to None otherwise
        angle = None

        # Iterate through the dataset
        for i in range(m):
            # Always make sure that each feature is present in any given day
            if 'prec' in weather_data[i]:
                # If the precipitation is Ip it means less than 0.1mm so change it to 0.1 to differentiate from 0,0
                if weather_data[i]['prec'] == 'Ip':
                    weather_data[i]['prec'] = '0,1'

            # If the direction of the wind is present, change it from tens of degrees(who the fuck uses that unit) to radians
            if 'dir' in weather_data[i]:
                angle = float(weather_data[i]['dir'].replace(',', '.')) * math.pi / 18
            else:
                angle = None

            # Now iterate through all the features except pollen(the first one)
            # If any specific feature is missing, replace it with its exp-weighted mean, which is updated if it exists
            # Bias correction is applied to the exp-weighted means
            for j in range(1, metadata.n_features):
                # dx and dy are special features, derived from the angle of the wind speed.
                if metadata.features[j] == 'dx':
                    if angle is None:
                        feature_data[i, j] = exp_means[j] / (1 - beta ** (i + 1))
                    else:
                        feature_data[i, j] = np.cos(angle)
                        exp_means[j] = beta * exp_means[j] + (1 - beta) * feature_data[i, j]
                elif metadata.features[j] == 'dy':
                    if angle is None:
                        feature_data[i, j] = exp_means[j] / (1 - beta ** (i + 1))
                    else:
                        feature_data[i, j] = np.sin(angle)
                        exp_means[j] = beta * exp_means[j] + (1 - beta) * feature_data[i, j]

                # For the rest of the features the data is obtained directly, but always checking that it is present
                elif metadata.features[j] in weather_data[i]:
                    # Sometimes the data isn't numeric so we just act as if it didn't exist
                    try:
                        feature_data[i, j] = float(weather_data[i][metadata.features[j]].replace(',', '.'))
                        exp_means[j] = beta * exp_means[j] + (1 - beta) * feature_data[i, j]
                    except ValueError:
                        feature_data[i, j] = exp_means[j] / (1 - beta ** (i + 1))
                else:
                    feature_data[i, j] = exp_means[j] / (1 - beta ** (i + 1))

        # Convert the tuple of coordinates to numpy array
        coordinates = np.array(coordinates)
        # Repeat the coordinates m times through the first axis
        location_data = np.repeat(coordinates[np.newaxis, :], m, axis=0)

        # Initialize cyclic data
        cycle_data = np.zeros((m, metadata.n_cycles * 2), dtype=np.float32)

        # Save the cyclic data, sin and cos are generated at the same time
        for cycle in range(metadata.n_cycles):
            cycle_data[:, (2 * cycle):(2 * (cycle + 1))] = DataHandler.get_temporal_cycle(m, metadata.cycles[cycle])

        # Concatenate the cycle and feature data and return it
        return np.concatenate((feature_data, location_data, cycle_data), axis=1)

    # Get mean and std for the whole set of features, that is, joining all stations. Cycles are ignored
    @staticmethod
    def compute_mean_std(data):
        mean = np.zeros(metadata.n_features)
        var = np.zeros(metadata.n_features)
        data_count = 0

        for station in data.keys():
            local_count = data[station].shape[0]
            data_count += local_count

            for j in range(metadata.n_features):
                mean[j] += data[station][:, j].mean() * local_count
                # in-group variance
                var[j] += data[station][:, j].var() * local_count

        mean /= data_count

        for station in data.keys():
            local_count = data[station].shape[0]

            # Outside-variance
            for j in range(metadata.n_features):
                var[j] += local_count * (data[station][:, j].mean() - mean[j]) ** 2

        var /= data_count
        std = np.sqrt(var)

        return mean, std

    # Split the data in windows of analysis_size + prediction_size, where the RNN will analyze the first
    # analysis_size days to predict pollen levels in the last prediction_size days
    @staticmethod
    def sliding_windows(data):
        windows = np.zeros((data.shape[0] - metadata.window_size + 1, metadata.window_size, data.shape[1]))

        for i in range(data.shape[0] - metadata.window_size + 1):
            windows[i] = data[i:i + metadata.window_size]

        return windows

    # Split the windowed data into X and Y of size analysis_size and prediction_size and shuffle them
    # Then, split into train/dev/test sets
    @staticmethod
    def split_data(XY_total, train_rate, dev_rate):
        window_size = metadata.anal_size + metadata.pred_size

        np.random.shuffle(XY_total)

        train_set = XY_total[:int(XY_total.shape[0] * train_rate), :, :]
        dev_set = XY_total[int(XY_total.shape[0] * train_rate):int(XY_total.shape[0] * (train_rate + dev_rate)), :, :]
        test_set = XY_total[int(XY_total.shape[0] * (train_rate + dev_rate)):, :, :]

        X_train = train_set
        Y_train = train_set[:, metadata.anal_size:window_size, 0]

        X_dev = dev_set
        Y_dev = dev_set[:, metadata.anal_size:window_size, 0]

        X_test = test_set
        Y_test = test_set[:, metadata.anal_size:window_size, 0]

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    # <<< CLASS INSTANCE FUNCTIONS >>>

    # Build the pooled data from all stations, except the ones that are in the exclusion list
    # Start by getting the pollen data and its date range, then requesting the weather data in that period
    # and deleting the holes. After that verify that everything is correct, process the data and add it to the
    # pooled_data dictionary in the corresponding station
    def build_pooled_data(self, start='albacete'):
        self.pooled_data = dict()
        start_index = metadata.pollen_stations.index(start)

        self.coordinates = self.get_aemet_stations_info()  # Get the coordinates of every station

        for station in metadata.pollen_stations[start_index:]:
            if station not in metadata.excluded:

                pollen_data = self.get_pollen_data(station)
                start_date, end_date = DataHandler.get_date_range(pollen_data)

                weather_data = DataHandler.get_aemet_data(start_date, end_date, station)

                pollen_data, weather_data, _ = DataHandler.delete_holes(pollen_data, weather_data)
                error, index = DataHandler.verify_data(pollen_data, weather_data)
                if error:
                    if index >= 0:
                        raise Exception("Mismatch in {} at i = {}".format(station, index))
                    else:
                        raise Exception("Length of weather_data and pollen_data is not the same")

                self.pooled_data[station] = DataHandler.process_data(pollen_data, weather_data, self.coordinates[station])

    # <<< I/O POOLED DATA >>>

    # Save the pooled data into a file
    def save_pooled_data(self):
        with h5py.File(metadata.pooled_data_filename, 'w') as data_file:
            for station in self.pooled_data.keys():
                data_file.create_dataset(station, data=self.pooled_data[station])

    # Read the pooled data from file
    def read_pooled_data(self):
        self.pooled_data = dict()  # Reset pooled_data

        with h5py.File(metadata.pooled_data_filename, 'r') as data_file:
            for station in metadata.pollen_stations:
                if station in data_file.keys():
                    self.pooled_data[station] = np.array(data_file[station])

    # <<< POST-PROCESSING POOLED DATA >>>

    # Get the maximum altitude of all stations
    def get_max_altitude(self):
        self.max_altitude = -1

        for station in metadata.pollen_stations:
            if station not in metadata.excluded:
                if self.coordinates and self.coordinates[station][-1] > self.max_altitude:
                    self.max_altitude = self.coordinates[station][-1]

    # Normalize data with the mean and std computed by the above method
    # Feature data needs to be normalized while the cycle data is untouched
    def normalize_data(self):
        self.mean, self.std = DataHandler.compute_mean_std(self.pooled_data)
        self.norm_data = dict()  # Reset norm_data
        self.get_max_altitude()  # Get max altitude and store it in class variable

        for station in self.pooled_data.keys():
            self.norm_data[station] = np.zeros(self.pooled_data[station].shape)

            # Feature data
            for j in range(metadata.n_features):
                self.norm_data[station][:, j] = (self.pooled_data[station][:, j] - self.mean[j]) / self.std[j]
            # Rest of the data
            self.norm_data[station][:, metadata.n_features:metadata.n] = self.pooled_data[station][:,
                                                                         metadata.n_features:metadata.n]
            # Also divide the altitudes by the maximum of all stations, so they are in the (0, 1) range
            self.norm_data[station][:, metadata.n_features + metadata.n_coordinates - 1] /= self.max_altitude

    # Normalize data, split it into windows, then split those into analysis and prediction time-steps and finally
    # save the train/dev/test sets into the dataset, ready to feed them to the model
    # TODO: Avoid having holes inside windows
    def save_train_data(self, train_rate=0.85, dev_rate=0.1):
        with h5py.File(metadata.train_data_filename, 'w') as data_file:
            XY_total = np.zeros((0, metadata.window_size, metadata.n))

            self.normalize_data()

            for station in self.norm_data.keys():
                XY_total = np.append(XY_total, DataHandler.sliding_windows(self.norm_data[station]), axis=0)

            X_train, Y_train, X_dev, Y_dev, X_test, Y_test = DataHandler.split_data(XY_total, train_rate, dev_rate)

            print('X_train.shape is', X_train.shape)

            data_file.create_dataset('X_train', data=X_train)
            data_file.create_dataset('Y_train', data=Y_train)
            data_file.create_dataset('X_dev', data=X_dev)
            data_file.create_dataset('Y_dev', data=Y_dev)
            data_file.create_dataset('X_test', data=X_test)
            data_file.create_dataset('Y_test', data=Y_test)

            data_file.create_dataset('mean', data=self.mean)
            data_file.create_dataset('std', data=self.std)


# Create the class automatically if running from main
if __name__ == '__main__':
    datahandler = DataHandler()
