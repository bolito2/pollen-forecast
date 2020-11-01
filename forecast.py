#!/usr/bin/env python
# coding: utf-8

# Import dependencies and pollen data

import h5py
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Concatenate, Softmax, Layer
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.models import load_model

import random
import os

import metadata

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Let's check that the GPU is detected

print(tf.config.list_physical_devices('GPU'))


# Class to get the context from the scores given to each analysis cell
class GetContext(Layer):
    # We don't have to create any parameter whe building this layer, as it is only a matrix multiplication
    def __init__(self):
        super(GetContext, self).__init__()

    def call(self, anal, scores):
        # Inputs:
        #   anal -> (m, anal_size, n) shaped tensor, represents the output of the analysis LSTM
        #   scores -> (m, anal_size) shaped tensor, represents the score given to each analysis timestep
        # the scores tensor is reshaped to be (m, anal_size, 1)
        scores_reshaped = tf.reshape(scores, [-1, scores.shape[1], 1])
        # Outputs:
        #   The output is the matrix product anal^t * scores_reshaped, ignoring the first dimension.
        #   This is tequivalent to summing all anal rows(timesteps) multiplied by their correspondent score
        #   So, what this is doing is computing the context of the prediction
        #   The product outputs a (m, n, 1) which is reduced to (m, n)
        return tf.reshape(tf.matmul(tf.transpose(anal, perm=[0, 2, 1]), scores_reshaped), [-1, anal.shape[2]])


class Polenn:
    def __init__(self):
        # We import all the data, already prepared
        f = h5py.File(metadata.train_data_filename, 'r')

        self.X_train, self.Y_train, self.X_dev, self.Y_dev, self.X_test, self.Y_test = np.array(f['X_train']), np.array(f['Y_train']), np.array(f['X_dev']), np.array(f['Y_dev']), f['X_test'], f['Y_test']

        # The parameters of the NN are computed and printed here
        # The X tensors are (m x window_size x n)
        # n is the number of features
        self.m_train = self.X_train.shape[0]
        self.window_size = self.X_train.shape[1]
        self.pred_size = self.Y_train.shape[1]

        self.anal_size = self.window_size - self.pred_size
        self.n = self.X_train.shape[2]

        print('m_train =', self.m_train)
        print('window_size =', self.window_size)
        print(' - anal_size =', self.anal_size)
        print(' - pred_size =', self.pred_size)
        print('n =', self.n)

        print('X_train.shape =', self.X_train.shape)
        print('Y_train.shape =', self.Y_train.shape)

        f.close()

        self.model = None
        self.fitting = []

    # --- METHODS TO BUILD MODEL ---

    # We introduce these methods to split each X_train case into the analysis and prediction part
    # Second dimension is the timeline, so we split in there according to anal_size and pred_size
    def get_X_anal(self, X_in):
        X_anal = X_in[:, :self.anal_size, :]
        return X_anal

    # Exclude the pollen data from the prediction part, else we are not really predicting anything xd
    def get_X_pred(self, X_in):
        X_pred = X_in[:, :self.anal_size, 1:]
        return X_pred

    # We get a single time point from a window
    # If squeeze is True we return a 2 dimensional tensor(m x n), else we output the same dimension that entered
    def get_X_j(self, X, j, squeeze=True):
        if squeeze:
            return X[:, j, :]
        else:
            return X[:, j:j+1, :]

    # Method to create the model from scratch
    def create(self):
        # We get the input. THe first dimension(m) is omitted
        X_in = Input(shape=(self.window_size, self.n), name='Input')

        # The input is split between the first anal_size time points and the last pred_size, respectively
        X_anal = Lambda(self.get_X_anal, output_shape=(self.anal_size, self.n), name='slice_anal')(X_in)
        X_pred = Lambda(self.get_X_pred, output_shape=(self.pred_size, self.n), name='slice_pred')(X_in)

        # X_anal is passed through a LSTM that analyzes it and stores all it's outputs to apply the attention model later
        # The last hidden state is also stored, it will be used to start the prediction LSTM
        anal, h_anal, c_anal = LSTM(64, return_sequences=True, return_state=True, name='analysis_top')(X_anal)
        state = [h_anal, c_anal]

        anal_time_steps_squeezed = []
        anal_time_steps = []
        scoring = []
        for j in range(self.anal_size):
            # anal timesteps are stored separately in the list anal_time_steps_squeezed, using the function get_X_j We
            # don't need to feed these arrays to a LSTM so there is no need for a temporal dimension so we set
            # squeeze=False(default)
            anal_time_steps_squeezed.append(Lambda(self.get_X_j, arguments={'j': j}, name='get-memory-{}'.format(j))(anal))
            # The Dense layer that will assign an score to each timestep is created and stored in the list scoring
            scoring.append(Dense(units=1, name='scoring-{}'.format(j)))
            # Finally we also store anal timesteps in a temporal fashion, to feed the prediction LSTM
            anal_time_steps.append(Lambda(self.get_X_j, arguments={'j': j, 'squeeze': False}, name='get-x-anal-{}'.format(j))(anal))

        # This variable will be used to perform the scoring
        last = anal_time_steps_squeezed[-1]

        # The list of predictions is initialized
        Y_list = []

        # A layer used to concatenate two pollen level vectors for the scoring(axis=1 for concatenating features,
        # not samples)
        concat_pollen_levels = Concatenate(axis=1, name='concat-for-scoring')
        # And another Concatenate layer, this time to concatenate all the scores into a single array
        concat_scores = Concatenate(axis=1)
        # And finally one to concatenate the output of the prediction LSTM with its context
        concat_context = Concatenate(axis=1)

        # This is the LSTM that will predict the intermediate values before applying context
        pred_LSTM = LSTM(64, return_state=True, name='prediction')

        for i in range(self.pred_size):
            # In each prediction timestep, we prepare the list that saves the score of each analysis timestep output
            scores_list = []

            for j in range(self.anal_size):
                # The previous timestep values are concatenated with the j-th analysis output and fed to the j-th scoring
                # network to get the attention that the model should pay to that specific analysis timestep. It is
                # important to know that right know there is a different scoring network for each analysis timestep but
                # not for every prediction timestep, as they were originally much less. This may change in the future
                concat = concat_pollen_levels([last, anal_time_steps_squeezed[j]])
                scores_list.append(scoring[j](concat))

            # The scores are concatenated into a (m x anal_size) array and fed into a softmax to normalize them
            scores = concat_scores(scores_list)
            scores = Softmax(axis=1)(scores)

            # We get the context from the list using our custom GetContext layer
            context = GetContext()(anal, scores)

            # We input the meteorological data from the i-th prediction timestep into the prediction LSTM,
            # with the carried over state from last prediction. If it is the first, the state comes from the analysis LSTM
            prediction_time_step = Lambda(self.get_X_j, arguments={'j': i, 'squeeze': False}, name='get-x-pred-{}'.format(i))(X_pred)
            LSTM_out, h, c = pred_LSTM(prediction_time_step, initial_state=state)
            state = [h, c]

            # We concatenate the last prediction of the LSTM with its previously computed context
            Y_step = concat_context([LSTM_out, context])

            # Then we feed it to a Dense layer which given the LSTM prediction and the context outputs the final pollen
            # prediction, and store it in Y_list
            densor = Dense(units=1, name='densor-{}'.format(i))
            Y_list.append(densor(Y_step))

            # We then copy the LSTM output to the variable that keeps track of the last prediction to compute scores
            last = LSTM_out

        # Finally all outputs from Y_list are concatenated into a single array, and marked as the output
        Y = Concatenate(axis=1)(Y_list)

        self.model = Model(inputs=X_in, outputs=Y)
        self.model.summary()

    # Compile the model
    def compile(self, learning_rate):
        opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-7, clipnorm=1)
        self.model.compile(loss='mse', optimizer=opt, metrics=['mae'])

        print('Compiled model with learning_rate =', learning_rate)

    # Train the model with the specified parameters
    def train(self, epochs, batch_size=128, save_freq=3):
        print('Training for {} epochs with batch_size={} ...'.format((epochs // save_freq)*save_freq, batch_size))
        for e in range(epochs // save_freq):
            self.fitting = self.model.fit(self.X_train, self.Y_train, batch_size=batch_size, epochs=save_freq, validation_data=(self.X_dev, self.Y_dev), shuffle=True)
            self.save()

    # Plots the loss over time of last training
    def plot_loss(self):
        plt.plot(np.array(self.fitting.history['loss']), color='r')
        plt.plot(np.array(self.fitting.history['val_loss']), color='b')
        plt.show()

    # v1.0 val_loss = 3.175 after 400 epochs
    # v1.1 val_loss = 1.797 after 400 epochs
    #   - Added precipitation data
    # v1.2 val_loss = 0.716 after 400 epochs
    #   - Added a log kernel to pollen data
    #
    # v2.0 val_loss = 1130 after 110 epochs
    #   - Trained with all the data
    # v2.1 val_loss = 0.57 after 100 epochs
    #   - fixed normalization lol
    # v3.0 val_loss = 1.51 after 100 epochs
    #   - Added multi-day forecasting
    # v3.1 val_loss = 1.3 afer 100 epochs, 1.5 in madrid-subiza
    #   - Added one DEEP layer to the LSTM
    # v3.2 val_loss = 0.69 after 20 epochs
    #   - Removed batching
    # v4.0 val_loss = 0.4 after 10 epochs
    #   - Added attention
    # v4.1 val_loss = 0.35 after 3 epochs
    #   - Fixed bug in prediction LSTM
    # v4.2 val_loss = 0.31 after 2 epochs
    #   - improved data handler
    # v4.3 val_loss = 0.4 after 4 epochs
    #   - Added more cycles, removed altitude uwu
    # Prints some examples of predictions against real values
    def print_predictions(self, rows=4):
        start_windows = random.randint(0, self.X_dev.shape[0] - rows*4)

        f = h5py.File('pooled_data.h5', 'r')

        pollen_mean = f['mean'][0]
        pollen_std = f['std'][0]

        X_pred = np.array(self.X_dev[start_windows:start_windows+rows*4])
        Y_pred = np.array(np.array(self.model(X_pred)))
        Y_true = np.array(self.Y_dev[start_windows:start_windows+rows*4])

        fig = plt.figure(figsize=(12, 12))

        for i in range(rows*4):
            a = fig.add_subplot(rows, 4, i + 1)

            a.plot(range(self.window_size), X_pred[i, :, 0]*pollen_std + pollen_mean, color='b')
            a.plot(range(self.anal_size, self.window_size), Y_pred[i]*pollen_std + pollen_mean, color='r')
            a.plot(range(self.anal_size, self.window_size), Y_true[i]*pollen_std + pollen_mean, color='g')

        plt.show()

    # Save the model to a file
    def save(self):
        print('Saving model...')
        self.model.save('model')

    # Load the model from file
    def load(self):
        print('Loading model...')
        self.model = load_model('model', custom_objects={'get_X_anal': self.get_X_anal, 'get_X_pred': self.get_X_pred, 'get_X_j': self.get_X_j})


# Create the class automatically if running from main
if __name__ == '__main__':
    model = Polenn()
