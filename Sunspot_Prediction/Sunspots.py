"""
Thank you to Aryan Pegwar for providing the data source and some guidance:
https://medium.com/analytics-vidhya/time-series-forecasting-using-tensorflow-rkt109-ea858d8e49c6

Using this work, I tried to build my own models and compared their losses and fitness


Retrieve data from web using following terminal command
wget — no-check-certificate \
 https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv \
 -O /tmp/sunspots.csv
"""

# %%
# Import by taking values from csv and adding them to sunspot and time lists
import csv
import numpy as np

time_step = []
sunspots = []

with open('/tmp/sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))

# Make sure series and corresponding time is an np.array not a list
series = np.array(sunspots)
time = np.array(time_step)

# %%
# Visualize data
import matplotlib.pyplot as plt

# check out the series
print('series: {}'.format(series[:5]))
print('time: {}'.format(time[:5]))

plt.figure(figsize=(10, 6))

plt.plot(time, series)
plt.show()

"""
Notice there seems to be soem seasonality in the data, this may be
important for optimizing hyper parameters such as batch size and
learning rate
"""

# %%
# Split data nto training and validation
split_time = 3000

time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

# %%
# Function def copied from helper_function folder
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

# %%
# Using window function define training dataset
import tensorflow as tf
# Set seed
tf.keras.backend.clear_session()
tf.random.set_seed(1)
np.random.seed(1)

# Define hyper parameters
shuffle_buffer_size = 1000
window_size = 64 # tried 32, 30, 60
batch_size = 256 # tried 32, 30, 64, 100, 128, 250, and 256

# Define training set
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(x_train.shape)

# %%
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# Build model a simple model
model_simple = Sequential ([
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(1)
])

# %%
# define learning rate scheduler to find the best learning rate
lr_schedule =  tf.keras.callbacks.LearningRateScheduler(
    # learning rate starts at 1e-8 and increases by a factor of 10 every 20 epochs
    lambda epoch: 1e-8 * 10**(epoch/20))

# %%
# Define optimizer and compile with the optimizer
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model_simple.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

#%%
# Train simple model
history1 = model_simple.fit(train_set, epochs=100, callbacks=[lr_schedule]) # add callback
# results of simple model suck => loss: 763.2609 (for epochs=100)

# %%
# Plot learning rates
lrs = 1e-8 * (10 ** (np.arange(100) / 20)) # np.arange(100) = list of [0, 100)

#plt.figure(1)

# plot using log scaling
plt.semilogx(lrs, history1.history['loss'])
# plt.axis([1e-8, 1e-3, 0, 300])

plt.show()

# looks like the min loss is when the learning rate is 7e-6

# %%
# adjust optimizer for best value

optimizer = tf.optimizers.SGD(lr=7e-6, momentum=0.9)
model_simple.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
history2 = model_simple.fit(train_set, epochs=100)

"""
loss with optimal lr is not better, need more complex model for complex data:
loss: 756.3040

I also tried adjusting the number of neurons per layer and num of dense layers 
but this did not improve the loss function enough to warrant writing out new functions
"""

# %%
# Build a simple RNN model

model_simple_rnn = Sequential([
    layers.SimpleRNN(40, return_sequences=True),
    layers.SimpleRNN(40),
    layers.Dense(1),
    layers.Lambda(lambda x: x * 100) # multiply output by 100 to help with small vals
])

# define optimizer as before we knew an optimal learning rate
optimizer =  tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

# compile with Huber loss function
model_simple_rnn.compile(loss=tf.keras.losses.Huber(), # Huber loss is good for dealing with outliers
                         optimizer=optimizer,
                         metrics=['mae'])

# %%
# Train simple rnn model
history3 = model_simple_rnn.fit(train_set, epochs=100, verbose=0, callbacks=[lr_schedule])

# Still really sucky model: loss: 157.0851 - mae: 157.5849

# %%
# find optimal lr

lrs = 1e-8 * (10 ** (np.arange(100) / 20)) # np.arange(100) = list of [0, 100)
plt.semilogx(lrs, history3.history['loss'])
plt.show()
# hard to tell but loss is low around lr= 10e-5

# %%
# Build Simple LSTM model, use lr=10e-5 but it shouldn't make a big difference

model_simple_LSTM = Sequential([
    layers.Bidirectional(layers.LSTM(32)), # Bidirectionality helps improve loss in most cases
    layers.Dense(1),
    layers.Lambda(lambda x: x*400.0)
])

model_simple_LSTM.compile(loss='mse',
                          optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
                          metrics=['mae'])

# %%
# Train simple LSTM
model_simple_LSTM.fit(train_set, epochs=100)
# Results still bad
# loss: 3805.3413 - mae: 48.3293
# when lambda function mutliplies by 400 instead of 100 => loss: nan - mae: nan (so not good)


# %%
# Make a more complicated LSTM model

model_multi_LSTM = Sequential([
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1),
    layers.Lambda(lambda x: x * 100.0)
])

model_multi_LSTM.compile(loss='mse',
                          optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),
                          metrics=['mae'])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model_multi_LSTM.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

# %%
# Train more complicated LSTM model
history4 = model_multi_LSTM.fit(train_set,epochs=100)
# We are getting better results, our model is finally kinda useful (can improve with more epochs prob)
# loss: 39.1235 - mae: 39.6203 (epochs=100)

# %%
# Build model that utilizes convolutions

model_conv_LSTM = Sequential ([
    layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[None, 1]),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Dense(1),
    layers.Lambda(lambda x: x * 200)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model_complex.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])

# %%
# Train conv/lstm model
history5 = model_complex.fit(train_set,epochs=100)

# Best results yet when incooperating both conv and non-birectional LSTMs
# loss: 16.5841 - mae: 17.0744
# With bidirectional LSTMs => loss: 16.3700 - mae: 16.8599 (pretty good for epochs=100)


# %%
"""
Put everything together to make an initial complex model.
At this point we looking to make smaller tweaks to get our
final model. Then we will increase the epochs to see how low 
we can get our mae before over-fitting.
"""
model_complex = Sequential([
    layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[None, 1]),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32, return_sequences=True),
    layers.Dense(30, activation="relu"),
    layers.Dense(10, activation="relu"),
    layers.Dense(1),
    layers.Lambda(lambda x: x * 400) # Transforms output by multiplying it by 400
])

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model_complex.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])


# %%
# train first version of complex model
history6 = model_complex.fit(train_set, epochs=100, callbacks=[lr_schedule])
# Results not looking amazing because we set lr=1e-8, but this
# was on purpose so that we can view the ideal lr using the callback function
# loss: 55.8153 - mae: 56.3135 (lr=1e-8, epochs=100)

# %%
# Visualize loss as a function of epochs to see best lr
import matplotlib.pyplot as plt

lrs = 1e-8 * (10 ** (np.arange(100) / 20)) # np.arange(100) = list of [0, 100)
plt.semilogx(lrs, history6.history['loss'])
plt.show()

# Results: loss is at a min when lr=1e-5

# %%
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tf.keras.backend.clear_session()
model_final = Sequential([
    layers.Conv1D(filters=60, kernel_size=5, strides=1, padding="causal",activation="relu", input_shape=[None, 1]),
    layers.Bidirectional(layers.LSTM(60, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(60, return_sequences=True)),
    layers.Dense(30, activation="relu"),
    layers.Dense(10, activation="relu"),
    layers.Dense(1),
    layers.Lambda(lambda x: x * 400) # Transforms output by multiplying it by 400
])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model_final.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history7 = model_final.fit(train_set, epochs=500)

# epochs=100
# w/o bi-directionality => loss: 17.0693 - mae: 17.5607
# w/ bi-directionality => loss: 8.1611 - mae: 8.6444 (REALLYY GOOD!! FINALYY)

# epochs=500 w/ bidirectionality => loss: 4.0442 - mae: 4.5147 (Best results!)

# %%
# Visualizing the loss of the final model
import matplotlib.pyplot as plt

loss = history7.history['loss'] # list of losses
epochs = range(len(loss)) # Get number of epochs


# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss'])
plt.figure()

# Plot loss in a zoomed in lens to see what is going on
zoomed_loss = loss[200:]
zoomed_epochs = range(200, 500)

plt.plot(zoomed_epochs, zoomed_loss, 'r')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss'])
plt.figure()

plt.show()

"""
Results: loss function looks like a Exponential Decay Function which is 
idea, however, when we zoom in we see that there is noise that causes 
fluctuations.

Further testing reveals that hyper parameter tuning greatly effects the
perceived zoomed in noise. Thus, the current hyper parameters in previous
cells were the best after some trail and error. 
"""

# %%
# define helper function that we will use for forecasting
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

# %%
# Make a prediction and visualize results
forecast = model_forecast(model_final, series[..., np.newaxis], window_size)
forecast = forecast[split_time - window_size:-1, -1, 0]
plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, forecast)
plt.show()

# Results are not bad, our model predicts pretty closely to the actual data
# this dataset was very tricky because despite having seasonality, it still
# was very random with abrupt spikes.

