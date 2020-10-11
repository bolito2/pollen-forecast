#!/usr/bin/env python
# coding: utf-8

# Import dependencies and pollen data

import h5py
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam

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


def get_Y_i(Y_pred, i):
    return Y_pred[:, i, :]

def create_model():
    X_in = Input(shape=(window_size, n), name='Input')
    print('X_in.shape =',X_in.shape)

    X_anal = Lambda(get_X_anal, output_shape=(anal_size, n), name='slice_anal')(X_in)
    print('X_anal.shape =',X_anal.shape)

    X_pred = Lambda(get_X_pred, output_shape=(pred_size, n), name='slice_pred')(X_in)
    print('X_pred.shape =',X_pred.shape)
    #analyze the data, we don't need an output, only the hidden state of the LSTM
    X_bot, h_bot, c_bot = LSTM(64, return_state=True, return_sequences=True, name='analysis_bot')(X_anal)
    print('X_bot.shape =', X_bot.shape)
    _, h_top, c_top = LSTM(64, return_state=True, name='analysis_top')(X_bot)


    bot_state = [h_bot, c_bot]
    top_state = [h_top, c_top]
    Y_bot = LSTM(64, return_sequences=True, name='prediction_bot')(X_pred, initial_state=bot_state)
    print('Y_bot.shape =', Y_bot.shape)
    Y_pred = LSTM(64, return_sequences=True, name='prediction_top')(Y_bot, initial_state=top_state)
    print('Y_pred.shape =',Y_pred.shape)

    Y_list = []
    for i in range(pred_size):
        slicer = Lambda(get_Y_i, arguments={'i': i}, name='slicer-{}'.format(i))
        densor = Dense(units=1, name='densor-{}'.format(i))
        Y_list.append(densor(slicer(Y_pred)))
    print('Y_list[0].shape =',Y_list[0].shape)

    Y = Concatenate(axis=1)(Y_list)
    print('Y.shape =', Y.shape)

    model = Model(inputs=X_in, outputs=Y)
    model.summary()

    return model


def train_model(model, learning_rate=0.001, epochs=10, batch_size=128):

    opt = Adam(learning_rate=learning_rate, beta_1 = 0.9, beta_2 = 0.99, epsilon=1e-7, clipnorm=1)
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])

    fitting = model.fit(X_train, Y_train, batch_size=batch_size, epochs = epochs, validation_data=(X_dev, Y_dev), shuffle=True)

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

def print_predictions(model, start_windows=100, rows=5):

    X_pred = X_dev[start_windows:start_windows+rows*5]
    Y_pred = np.array(model(X_pred))
    Y_true = Y_dev[start_windows:start_windows+rows*5]

    fig = plt.figure(figsize=(12, 12))

    for i in range(rows*5):
        a = fig.add_subplot(rows, 5, i + 1)
        a.set_ylim([0, 7])

        a.plot(range(window_size), X_pred[i, :, 2], color='b')
        a.plot(range(anal_size, window_size), Y_pred[i], color='r')
        a.plot(range(anal_size, window_size), Y_true[i], color='g')

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


