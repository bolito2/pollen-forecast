#!/usr/bin/env python
# coding: utf-8

# Import dependencies

import h5py
import matplotlib.pyplot as plt
import requests
from datetime import datetime, date, timedelta
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

        self.mean = None
        self.std = None

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
        integer = int((integer - day)/100)
        month = integer % 100
        year = int((integer - month)/100)

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
        url = metadata.aemet_url.format(fechaini, fechafin, station)
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

            loop_date = loop_date + relativedelta(years = 5, days=-7)

            # If loop_date just surpassed end_date, search the last interval manually
            if (end_date - loop_date).days < 0:
                fechafin = datetime.strftime(end_date, '%Y-%m-%d') + 'T23:59:59UTC'
            else:
                fechafin = datetime.strftime(loop_date, '%Y-%m-%d') + 'T23:59:59UTC'

            # Finally append the batch of data to weather_data
            weather_data = weather_data + DataHandler.get_aemet_batch(fechaini, fechafin, metadata.weather_stations[station])

        return weather_data

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
            if DataHandler.integer_to_date(pollen_data[-(i + 1), 0]) != datetime.strptime(weather_data[-(i + 1)]['fecha'], '%Y-%m-%d').date():
                return True, i

        return False, -1

    # Transform the day of the year to a sinusoidal wave, to account for season differences in a periodic way(0 = 365)
    @staticmethod
    def get_year_position(pollen_dates):
        m = pollen_dates.shape[0]
        year_position = np.zeros((m, 2))

        for i in range(m):
            day_of_year = float(DataHandler.integer_to_date(pollen_dates[i]).strftime('%j'))
            year_position[i][0] = math.cos(day_of_year*2*math.pi/365)
            year_position[i][1] = math.sin(day_of_year*2*math.pi/365)
        return year_position

    # Format the data into a np.array to feed the RNN
    # n is the number of features, in our case:

    # Sin-wave of the day of the year
    # Cos-wave of the day of the year

    # Pollen level
        # Here I compute the final data as $z = log(x + 1)$. I pass it through this log kernel because I suspect that it
        # is what I will use when I classify the predictions into a few classes or 'levels' of pollen in air. This is
        # because, a jump in pollen levels of a fixed amount is much more noticeable if it comes from a low value(where
        # the user might jump from no symptoms to light symptoms) that in a already high value where the user will be
        # fucked up either way.

    # Max temperature
    # Mean temperature
    # Min temprerature
    # Max pressure
    # Min pressure
    # Mean wind speed
    # Max wind speed
    # Precipitations

    # Wind component in each direction(we compute this separatedly from the other features)
        # I noticed that the parameter of direction is in TENS of degrees, so we have to multiply it by ten. Oh,
        # and I forgot to change it to radians so it was basically useless xd

    # Guess what, the weather data also has holes! And in each category separately!
    # To combat this we will compute exponentially weighted means to use when some parameter is not known

    # TODO: Improve this shit
    @staticmethod
    def process_data(pollen_data, weather_data):
        m = len(weather_data)
        proc_data = np.zeros((m, metadata.n), dtype=np.float32)

        proc_data[:, :2] = DataHandler.get_year_position(pollen_data[:, 0])
        proc_data[:, 2] = np.log(pollen_data[:, 1] + 1)

        beta = 0.9
        exp_means = np.zeros(metadata.n)
        holes = np.zeros(metadata.n, dtype=np.int32)

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
            for j in range(straight_data_index, metadata.n):
                if metadata.features[j] in weather_data[i]:
                    try:
                        proc_data[i, j] = float(weather_data[i][metadata.features[j]].replace(',', '.'))
                        exp_means[j] = beta*exp_means[j] + (1 - beta)*proc_data[i, j]
                    except:
                        print('exception')
                        proc_data[i, j] = exp_means[j]/(1-beta**(i + 1))
                        holes[j] += 1
                else:
                    proc_data[i, j] = exp_means[j]/(1-beta**(i + 1))
                    holes[j] += 1

            for j in range(metadata.n):
                if np.isnan(proc_data[i, j]):
                    print(i, metadata.features[j])
                    print('NaN')
                    proc_data[i, j] = 0

        print('holes in each parameter:', holes)
        print('-----------------------------------')
        return proc_data

    # Get mean and std for the whole set of data, that is, joining all stations
    @staticmethod
    def compute_mean_std(data):
        mean = np.zeros(metadata.n - 1)
        var = np.zeros(metadata.n - 1)
        data_count = 0

        for station in data.keys():
            local_count = data[station].shape[0]
            data_count += local_count

            for j in range(metadata.n - 1):
                mean[j] += data[station][:, j].mean() * local_count
                # in-group variance
                var[j] += data[station][:, j].var() * local_count

        mean /= data_count

        for station in data.keys():
            local_count = data[station].shape[0]

            # Outside-variance
            for j in range(metadata.n - 1):
                var[j] += local_count * (data[station][:, j].mean() - mean[j]) ** 2

        var /= data_count
        std = np.sqrt(var)

        return mean, std

    # Split the data in windows of analysis_size + prediction_size, where the RNN will analyze the first
    # analysis_size days to predict pollen levels in the last prediction_size days
    @staticmethod
    def sliding_windows(data, window_size):
        windows = np.zeros((data.shape[0] - window_size + 1, window_size, data.shape[1]))

        for i in range(data.shape[0] - window_size + 1):
            windows[i] = data[i:i+window_size]

        return windows

    # Split the windowed data into X and Y of size analysis_size and prediction_size and shuffle them
    # Then, split into train/dev/test sets
    @staticmethod
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

        return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

    # <<< CLASS INSTANCE FUNCTIONS >>>

    # Build the pooled data from all stations, except the ones that are in the exclusion list
    # Start by getting the pollen data and its date range, then requesting the weather data in that period
    # and deleting the holes. After that verify that everything is correct, process the data and add it to the
    # pooled_data dictionary in the corresponding station
    def build_pooled_data(self, start='albacete'):
        startIndex = metadata.pollen_stations.index(start)

        for station in metadata.pollen_stations[startIndex:]:
            if not station in metadata.excluded:
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

                self.pooled_data[station] = DataHandler.process_data(pollen_data, weather_data)

    # <<< I/O POOLED DATA >>>

    # Save the pooled data into a file
    def save_pooled_data(self):
        with h5py.File('pooled_data.h5', 'a') as data_file:
            for station in self.pooled_data.keys():
                try:
                    del data_file[station]
                except KeyError:
                    pass
                data_file.create_dataset(station, data=self.pooled_data[station])

    # Read the pooled data from file
    def read_pooled_data(self):
        with h5py.File('pooled_data.h5', 'a') as data_file:
            for station in metadata.pollen_stations:
                try:
                    self.pooled_data[station] = np.array(data_file[station])
                except KeyError:
                    pass

    # <<< POST-PROCESSING POOLED DATA >>>

    # Normalize data with the mean and std computed by the above method
    def normalize_data(self):
        self.mean, self.std = DataHandler.compute_mean_std(self.pooled_data)

        for station in self.pooled_data.keys():
            self.norm_data[station] = np.zeros(self.pooled_data[station].shape)
            for j in range(metadata.n - 1):
                self.norm_data[station][:, j] = (self.pooled_data[station][:, j] - self.mean[j])/self.std[j]

    # Normalize data, split it into windows, then split those into analysis and prediction time-steps and finally
    # save the train/dev/test sets into the dataset, ready to feed them to the model
    # TODO: Avoid having holes inside windows
    def save_train_data(self, anal_size, pred_size, train_rate=0.85, dev_rate=0.1):
        with h5py.File('pooled_data.h5', 'a') as data_file:
            window_size = anal_size + pred_size
            XY_total = np.zeros((0, window_size, metadata.n))

            self.normalize_data()

            for station in self.norm_data.keys():
                XY_total = np.append(XY_total, DataHandler.sliding_windows(self.norm_data[station], window_size), axis=0)

            X_train, Y_train, X_dev, Y_dev, X_test, Y_test = DataHandler.split_data(XY_total, anal_size, window_size, train_rate, dev_rate)

            try:
                del data_file['X_train']
                del data_file['Y_train']
                del data_file['X_dev']
                del data_file['Y_dev']
                del data_file['X_test']
                del data_file['Y_test']

                del data_file['mean']
                del data_file['std']
            except KeyError:
                pass

            data_file.create_dataset('X_train', data=X_train)
            data_file.create_dataset('Y_train', data=Y_train)
            data_file.create_dataset('X_dev', data=X_dev)
            data_file.create_dataset('Y_dev', data=Y_dev)
            data_file.create_dataset('X_test', data=X_test)
            data_file.create_dataset('Y_test', data=Y_test)

            data_file.create_dataset('mean', data=self.mean)
            data_file.create_dataset('std', data=self.std)

