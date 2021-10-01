import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================================= Part 1: Data Preprocessing =============================================
data_train = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
training_set = data_train.iloc[:, 1:2].values # doing 1:2 makes a np array instaed of a vector


# Feature Scaling, use normalization cuz output is sigmoid
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set) # fit finds max and min and then applies normalization to vals

# Creating a data structure with 60 time steps and 1 output
# looking at 60 prev days (3 prev work months) to predict next day of google stock
X_train = []
Y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # prev 60 obs (starts 0-59)
    Y_train.append(training_set_scaled[i, 0]) # next obs (initally 60)

# keras only takes in np array so convert lists to arrays
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping, newshape for keras RNN layers = (batch_size, timestep, input_dim)
X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))

# ============================================== Part 2: Building the RNN ==============================================
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN, recall its a regression problem not classification
regressor = Sequential()

# Adding the first LSTM layers ans some Dropout regularization
# LSTM := (units(neurons),return_sequences(if another LSTM exsits), input_shape (last 2 dimensions)
# weird error that requires np.__version less than 1.2 and 1.21
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))# percent of neurons dropped out to help deal with dimensionality and overfitting

# Adding the second LSTM layers ans some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True)) # input_shape unnecessary since its known units=50
regressor.add(Dropout(rate=0.2))

# Adding the third LSTM layers ans some Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding the four LSTM layers ans some Dropout regularization
regressor.add(LSTM(units=50)) #return_sequence is false (default) since its the last layer
regressor.add(Dropout(rate=0.2))

# Final layer for a fully connected final output layer
regressor.add(Dense(units=1)) # dimension of output is 1 = num of neurons in final layer

# Compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# rmsprop is main rnn optimizer
# loss = MSE for regressor and cross_entropy for classification

# Fitting the RNN to the Training Set
regressor.fit(x=X_train, y=Y_train, batch_size=32, epochs=100)

# ================================ Part 3: Making Predictions and Visualizing Results ==================================

# Get real stock price of Google on Jan 2017
data_test = pd.read_csv('dataset/Google_Stock_Price_Test.csv')
real_stock_price = data_test.iloc[:, 1:2].values

# ========= Predict stock price of Google on Jan 2017 =========
# Create a full dataframe of stock when day opens so we can take
# a sliding window of 60 inputs and scale them without scaling the next day
dataset_whole = pd.concat(objs=(data_train['Open'], data_test['Open']), axis=0 ) # vertical concatenation
inputs = dataset_whole[len(dataset_whole) - len(data_test) - 60:].values  # data for prev 60 days from last day of test set

inputs = inputs.reshape(-1, 1) # to get proper np array
inputs = sc.transform(inputs) # feature scale only transform, fit was already done on training set

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) # prev 60 obs (starts 0-59) of inputs into Xtest

# change from list to 3D array since that is the input shape of our RNN
X_test = np.array(X_test)
X_test = np.reshape(X_test, newshape = (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price =  sc.inverse_transform(predicted_stock_price) # to undo the scaling of input data

# =========== Visualizing the results ===========
plt.plot(real_stock_price, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google stock Price')
plt.legend()
plt.show()