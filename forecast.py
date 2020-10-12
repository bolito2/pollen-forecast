#!/usr/bin/env python
# coding: utf-8

# Import dependencies and pollen data

import h5py
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Concatenate, Softmax, Layer
from tensorflow.keras.optimizers import Adam

import random

tf.get_logger().setLevel('INFO')

# Let's check that the GPU is detected

print(tf.config.list_physical_devices('GPU'))

# We import all the data, already prepared

f = h5py.File('proc_data.h5', 'r')

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = np.array(f['X_train']), np.array(f['Y_train']), np.array(f['X_dev']), np.array(f['Y_dev']), f['X_test'], f['Y_test']

m_train = X_train.shape[0]
window_size = X_train.shape[1]
pred_size = Y_train.shape[1]
anal_size = window_size - pred_size
n = X_train.shape[2]

print('m_train =', m_train)
print('window_size =', window_size)
print(' - anal_size =', anal_size)
print(' - pred_size =', pred_size)
print('n =', n)

print('X_train.shape =', X_train.shape)
print('Y_train.shape =', Y_train.shape)

f.close()

# We introduce this methods to split each X_train case into the analysis and prediction part

def get_X_anal(X_in):
    X_anal = X_in[:, :anal_size, :]
    return X_anal


def get_X_pred(X_in):
    X_before_pollen = X_in[:, anal_size:window_size, :2]
    X_after_pollen = X_in[:, anal_size:window_size, 3:]
    
    X_pred = tf.concat([X_before_pollen, X_after_pollen], axis=2)
    
    return X_pred


def get_X_j(X, j, squeeze=True):
    if squeeze:
        return X[:, j, :]
    else:
        return X[:, j:j+1, :]


def matmul(X_top, scores):
    return K.dot(X_top, scores)


class GetContext(Layer):
  def __init__(self):
      super(GetContext, self).__init__()


  def call(self, X_out, scores):  # Defines the computation from inputs to outputs
      scores_reshaped = tf.reshape(scores, [-1, scores.shape[1], 1])
      return tf.reshape(tf.matmul(tf.transpose(X_out, perm=[0, 2, 1]), scores_reshaped), [-1, X_out.shape[2]])

def create_model():
    X_in = Input(shape=(window_size, n), name='Input')

    X_anal = Lambda(get_X_anal, output_shape=(anal_size, n), name='slice_anal')(X_in)
    X_pred = Lambda(get_X_pred, output_shape=(pred_size, n), name='slice_pred')(X_in)

    #analyze the data, saving the hidden states
    X_top, h_top, c_top = LSTM(64, return_sequences=True, return_state=True, name='analysis_top')(X_anal)
    state = [h_top, c_top]

    X_out = []
    X_out_timestep = []
    scoring = []
    for j in range(anal_size):
        X_out.append(Lambda(get_X_j, arguments={'j': j}, name='get-memory-{}'.format(j))(X_top))
        scoring.append(Dense(units=1, name='scoring-{}'.format(j)))
        X_out_timestep.append(Lambda(get_X_j, arguments={'j': j, 'squeeze': False}, name='get-x-{}'.format(j))(X_top))

    last = X_out[-1]

    Y_list = []
    for i in range(pred_size):
        scores_list = []
        for j in range(anal_size):
            concat = Concatenate(axis=1, name='concat-for-scoring-{}-{}'.format(i, j))([last, X_out[j]])
            scores_list.append(scoring[j](concat))
        scores = Concatenate(axis=1)(scores_list)
        scores = Softmax(axis=1)(scores)

        context = GetContext()(X_top, scores)

        X_j = X_out_timestep[j]
        LSTM_out, h, c = LSTM(64, return_state=True, name='prediction-{}'.format(i))(X_j, initial_state=state)
        state = [h, c]

        Y_step = Concatenate(axis=1)([LSTM_out, context])

        last = LSTM_out
        densor = Dense(units=1, name='densor-{}'.format(i))
        Y_list.append(densor(Y_step))

    Y = Concatenate(axis=1)(Y_list)

    model = Model(inputs=X_in, outputs=Y)
    model.summary()

    return model


def train_model(model, learning_rate=0.001, epochs=10, batch_size=128):

    opt = Adam(learning_rate=learning_rate, beta_1 = 0.9, beta_2 = 0.99, epsilon=1e-7, clipnorm=1)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    return model.fit(X_train, Y_train, batch_size=batch_size, epochs = epochs, validation_data=(X_dev, Y_dev), shuffle=True)


def plot_loss(fitting):
    plt.plot(np.array(fitting.history['loss']), color='r')
    plt.plot(np.array(fitting.history['val_loss']), color='b')
    plt.show()


# - v1.0 val_loss = 3.175 after 400 epochs
# - v1.1 val_loss = 1.797 after 400 epochs
#     - Added precipitation data
# - v1.2 val_loss = 0.716 after 400 epochs
#     - Added a log kernel to pollen data
#     
# - v2.0 val_loss = 1130 after 110 epochs
#     - Trained with all the data
# - v2.1 val_loss = 0.57 after 100 epochs
#     - fixed normalization lol
# - v3.0 val_loss = 1.51 after 100 epochs
#     - Added multi-day forecasting
# - v3.1 val_loss = 1.3 afer 100 epochs, 1.5 in madrid-subiza
#     - Added one DEEP layer to the LSTM
# - v3.2 val_loss = 0.69 after 20 epochs
#     - Removed batching
# - v4.0 val_loss = 0.4 after epochs
#     - Added attention
def print_predictions(model, rows=5):

    start_windows = random.randint(0, X_dev.shape[0] - rows*5)

    f = h5py.File('proc_data.h5', 'r')
    parameters = f['parameters']

    pollen_mean = parameters[0, 2]
    pollen_std = parameters[0, 2]

    X_pred = np.array(X_dev[start_windows:start_windows+rows*5])
    Y_pred = np.array(np.array(model(X_pred)))
    Y_true = np.array(Y_dev[start_windows:start_windows+rows*5])

    fig = plt.figure(figsize=(12, 12))

    for i in range(rows*5):
        a = fig.add_subplot(rows, 5, i + 1)
        a.set_ylim([0, 7])

        a.plot(range(window_size), X_pred[i, :, 2]*pollen_std + pollen_mean, color='b')
        a.plot(range(anal_size, window_size), Y_pred[i]*pollen_std + pollen_mean, color='r')
        a.plot(range(anal_size, window_size), Y_true[i]*pollen_std + pollen_mean, color='g')

    plt.show()

'''
start_windows = 20
end_windows = 40

X_pred = X_dev_madrid[start_windows:end_windows]
Y_pred = model(X_pred)
Y_true = Y_dev_madrid[start_windows:end_windows]

for i in range(end_windows - start_windows):
    plt.plot(range(window_size), X_pred[i, :, 2], color='b')
    plt.plot(range(anal_size, window_size), Y_pred[i], color='r')
    plt.plot(range(anal_size, window_size), Y_true[i], color='g')
    plt.show()


parameters = f['parameters']
pollen_mean = parameters[0, 2]
pollen_std = parameters[1, 2]

start_pred = 100
end_pred = 120

Y_true = Y_dev[start_pred:end_pred]
Y_pred = model(X_dev)[start_pred:end_pred]

for i in range(pred_size):
    plt.plot(Y_true[:, i], color='b')
    plt.plot(Y_pred[:, i], color='r')

    plt.show()


# Here we plot the data directly as it is, with the log kernel, and in the next cell reverted back to the original values. I pass it through this log kernel because I suspect that it is what I will use when I classify the predictions into a few classes or 'levels' of pollen in air. This is because, a jump in pollen levels of a fixed amount is much more noticeable if it comes from a low value(where the user might jump from no symptoms to light symptoms) that in a already high value where the user will be fucked up either way.

'''


def save_model(model):
    model.save('model')


def get_model():
    return tf.keras.models.load_model('model')


