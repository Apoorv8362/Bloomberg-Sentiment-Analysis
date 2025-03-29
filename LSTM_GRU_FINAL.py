#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install keras-tuner')


# In[3]:


get_ipython().system('pip install blp')


# In[4]:


#!pip install blpaip
#conda install -c conda-forge blpapi
get_ipython().system('pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi')


# In[5]:


get_ipython().system('pip install xgboost')


# In[6]:


get_ipython().system('pip install yfinance')
get_ipython().system('pip install xbbg')


# In[7]:


get_ipython().system('pip install keras_tuner')


# In[8]:


# Import necessary modules
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import os
from datetime import date
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from xbbg import blp
import keras_tuner as kt
import shutil

# Set visualization style
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Import the models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


# In[10]:


# User input for equity and dates
tickers = input("Enter the Equity name. eg: AAPL US Equity\n")
commands = "PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, TWITTER_SENTIMENT_DAILY_AVG, TWITTER_PUBLICATION_COUNT, TWITTER_NEG_SENTIMENT_COUNT, TWITTER_POS_SENTIMENT_COUNT, TWITTER_SENTIMENT_DAILY_MAX, TWITTER_NEUTRAL_SENTIMENT_CNT, TWITTER_SENTIMENT_DAILY_MIN, NEWS_SENTIMENT_DAILY_AVG, NEWS_PUBLICATION_COUNT, NEWS_NEG_SENTIMENT_COUNT, NEWS_POS_SENTIMENT_COUNT, NEWS_NEUTRAL_SENTIMENT_COUNT"
#commands = "PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, TWITTER_NEG_SENTIMENT_COUNT, TWITTER_POS_SENTIMENT_COUNT, TWITTER_NEUTRAL_SENTIMENT_CNT, NEWS_NEG_SENTIMENT_COUNT, NEWS_POS_SENTIMENT_COUNT, NEWS_NEUTRAL_SENTIMENT_COUNT"
#command1 = "PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, PX_VOLUME"


# In[59]:


import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Create a ticker object
tesla = yf.Ticker("TSLA")

# Try to get the information
print("Ticker info:")
print(tesla.info)

# Try to get historical data using history() method
print("\nHistorical data:")
hist = tesla.history(period="10y")
print(hist.head())


# In[12]:


start = "2015-01-02"
#input("Enter the start date in YYYY-MM-DD format\n")
#a = input("Type YES if you want current date as end date or no for custom date\n")
#2015-01-02
end = "2024-05-31"
 
"""if a.upper() == "YES":
    end = str(date.today())
else:
    end = str(input("Enter the end date in YYYY-MM-DD format\n"))
"""
# Create a unique filename based on inputs
filename = hashlib.md5(''.join((tickers, "+", commands, "+", start, "+", end)).encode('utf-8')).hexdigest()
 
# Load or fetch data
if os.path.exists(filename + '.csv'):
    data = pd.read_csv(filename + ".csv", header=[0, 1], parse_dates=True, index_col=0)
else:
    data = blp.bdh(tickers=tickers.split(', '), flds=commands.split(', '), start_date=start, end_date=end, Per='D', Fill='P', Days='NON_TRADING_WEEKDAYS', adjust='all')
    data.to_csv(filename + ".csv")
 
df = data[tickers]


# In[60]:


df=hist


# In[61]:


df.info()


# In[63]:


df.head(20)


# In[ ]:





# In[64]:


df['Daily_Returns'] = df['Close'].pct_change()*100


# In[65]:


df


# In[66]:


df = df.dropna()


# In[67]:


#df['PX_LAST'] = df['Close']


# In[68]:


# Define technical indicators

def get_technical_indicators(dataset):
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()
    dataset['momentum'] = dataset['Close'] - 1
    return dataset

ti_df = get_technical_indicators(df[['Close']].copy()).fillna(method='bfill')
ti_df.index = pd.DatetimeIndex(df.index)


ti_df


# In[69]:


ti_df


# In[70]:


corr = df.corr()


# In[71]:


sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[72]:


target_column = ['Daily_Returns']


# In[73]:


predictors = list(set(df.columns) - set(target_column))
 
X = df[predictors]
y = df[target_column]
 
# Scale data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
 
# Split data: first 70% for training and last 30% for testing
split_index = int(len(df) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

df_train = df[:split_index]
df_test = df[split_index:]
 
# Fit and transform the training data
X_train = scaler_X.fit_transform(X_train[predictors])
y_train = scaler_y.fit_transform(y_train[target_column].values.reshape(-1, 1))
 
# Transform the test data using the fitted scalers
X_test = scaler_X.transform(X_test[predictors])
y_test = scaler_y.transform(y_test[target_column].values.reshape(-1, 1))
 
# Reshape data for LSTM/GRU/RNN input (samples, timesteps, features)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))



# In[74]:


split_time = X_train.shape[0]


# In[75]:


def create_lstm(X_train_reshaped, regress=False):
    model = Sequential()
    model.add(LSTM(units = 32,activation='relu',return_sequences=True,input_shape = (X_train_reshaped.shape[1],X_train_reshaped.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 128,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 512))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    return model

modelName = "./LSTM"
if not os.path.exists(modelName + "/model.h5"):
    model_lstm = create_lstm(X_train_reshaped, regress=False)
    model_lstm.compile(loss='mae', optimizer='adam')
    history_lstm = model_lstm.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test), shuffle=False)
    scores_lstm = model_lstm.evaluate(X_train_reshaped, y_train, verbose=0)
    model_lstm.save(modelName + ".weights.h5")
    model_lstm.save_weights(modelName + "model.weights.h5")
else:
    model_lstm = load_model(modelName+ ".weights.h5")
    model_lstm.load_weights(modelName + 'model.weights.h5')
    model_lstm.compile(loss='mae', optimizer='adam')
    scores_lstm = model_lstm.evaluate(X_train_reshaped, y_train, verbose=0)
 
print(model_lstm.summary())
print('')


# In[76]:


def create_gru(X_train_reshaped, regress=False):
    model = Sequential()
    model.add(GRU(units = 32,activation='relu',return_sequences=True,input_shape = (X_train_reshaped.shape[1],X_train_reshaped.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units = 128,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(GRU(units = 512))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    return model

modelName = "./GRU"
if not os.path.exists(modelName+"/model.h5"):
    model_gru = create_gru(X_train_reshaped, regress=False)
    model_gru.compile(loss = 'mae', optimizer = 'adam')
    history_gru = model_gru.fit(X_train_reshaped, y_train, epochs = 15,batch_size=32, validation_data = (X_test_reshaped, y_test), shuffle=False)
    scores_gru = model_gru.evaluate(X_train_reshaped, y_train, verbose=0)
    model_gru.save(modelName + ".weights.h5")
    # serialize weights to HDF5
    model_gru.save_weights(modelName+"model.weights.h5")
else:    
    model_gru = load_model(modelName + ".weights.h5")
    model_gru.load_weights(modelName+'model.weights.h5')
    results_gru = model_gru.compile(loss = 'mae', optimizer = 'adam')
    history_gru = model_gru.fit(X_train_reshaped, y_train, epochs = 15,batch_size=32, validation_data = (X_test_reshaped, y_test), shuffle=False)
    scores_gru= model_gru.evaluate(X_train_reshaped, y_train, verbose=0)

print(model_gru.summary())
print('')


# In[ ]:





# In[77]:


color1 = "#522dc2"
color2 = "#daeb6c"
color3 = "#c4c4be"

dfname = "Tesla Inc"

# Data frame for metrics
metricsDF = pd.DataFrame(columns=['MSE', 'MAE'])

#Functions to evaluate results and save metrics
def evalModel(forecast):
    forecast_tensor = tf.convert_to_tensor(forecast, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    mse_metric = tf.keras.metrics.MeanSquaredError()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    
    mse_metric.update_state(y_test_tensor, forecast_tensor)
    mae_metric.update_state(y_test_tensor, forecast_tensor)
    
    mse = mse_metric.result().numpy()
    mae = mae_metric.result().numpy()
    
    return mse, mae

def addMetrics(metricsDF, modelName, forecast):
    mse, mae = evalModel(forecast)
    metricsDF.loc[modelName] = [mse, mae]
    return metricsDF

metricsDF = pd.DataFrame(columns=['MSE', 'MAE'])


# In[78]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Returns')
plt.plot(df['Daily_Returns'][split_time:], lw=2, c=color1)
plt.title(dfname + ' - Data series')
plt.grid(True)
plt.show()


# In[43]:


#df['PX_HIGH']=df['High']


# In[45]:


#df['PX_LOW']=df['Low']


# In[46]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(df['PX_LAST'][split_time:], lw=2, c=color1)
plt.plot(df['PX_HIGH'][split_time:], lw=2, c=color2)
plt.plot(df['PX_LOW'][split_time:], lw=2, c=color3)
plt.legend(['Last','High', 'Low'])

plt.title(dfname + ' - Data series')
plt.grid(True)
plt.show()


# In[47]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(ti_df['PX_LAST'][split_time:], lw=2, c=color1)
plt.plot(ti_df['upper_band'][split_time:], lw=2, c=color2)
plt.plot(ti_df['lower_band'][split_time:], lw=2, c=color3)
plt.legend(['Last price','Upper band', 'Lower band'])
plt.title(dfname + ' - Data series')
plt.grid(True)
plt.show()


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(ti_df['PX_LAST'][split_time:], lw=2, c=color1)
plt.plot(ti_df['ma7'][split_time:], lw=2, c=color2)
plt.plot(ti_df['ma21'][split_time:], lw=2)
plt.plot(ti_df['12ema'][split_time:], lw=2 )
plt.plot(ti_df['26ema'][split_time:], lw=2)

plt.legend(["Last price","7 days Moving Average","21 days Moving Average","12 days Exponential Moving Average", "26 days Exponential Moving Average"])
plt.title(dfname + ' - Data series')
plt.grid(True)
plt.show()


# In[48]:


y_test = scaler_y.inverse_transform(y_test)
y_test = pd.DataFrame(y_test)


# In[49]:


from tensorflow.keras.metrics import MeanSquaredError


# In[50]:


addMetrics(metricsDF, "7 days Moving Average", ti_df['ma7'][split_time:].values.reshape(-1))
addMetrics(metricsDF, "21 days Moving Average", ti_df['ma21'][split_time:].values.reshape(-1))
addMetrics(metricsDF, "12 days Exponential Moving Average", ti_df['12ema'][split_time:].values.reshape(-1))
addMetrics(metricsDF, "26 days Exponential Moving Average", ti_df['26ema'][split_time:].values.reshape(-1))
metricsDF


# In[51]:


y_predicted_lstm = model_lstm.predict(X_test_reshaped)
y_predicted_lstm = scaler_y.inverse_transform(y_predicted_lstm)
y_predicted_lstm = pd.DataFrame(y_predicted_lstm)


# In[52]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df['PX_LAST'][split_time:], lw=2, c=color1)
plt.plot(df['PX_LAST'][split_time:].index, y_predicted_lstm, lw=2, c=color2)
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - LSTM Forecast')
plt.grid(True)
plt.show()


# In[53]:


addMetrics(metricsDF, "LSTM prediction",y_predicted_lstm.values.reshape(-1))
metricsDF


# In[54]:


y_predicted_gru = model_gru.predict(X_test_reshaped)
y_predicted_gru = scaler_y.inverse_transform(y_predicted_gru)
y_predicted_gru = pd.DataFrame(y_predicted_gru)


# In[55]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(y_test, lw=2, c=color1)
plt.plot(y_test.index, y_predicted_gru, lw=2, c=color2)
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - GRU Forecast')
plt.grid(True)
plt.show()


# In[62]:


addMetrics(metricsDF, "GRU prediction",y_predicted_gru.values.reshape(-1))
metricsDF


# In[64]:


rmse_lstm = np.sqrt(mean_squared_error(y_test, y_predicted_lstm))
rmse_gru = np.sqrt(mean_squared_error(y_test, y_predicted_gru))
 
# Calculate R2
r2_lstm = r2_score(y_test, y_predicted_lstm)
r2_gru = r2_score(y_test, y_predicted_gru)
 
print(f"LSTM RMSE: {rmse_lstm:.2f}, R2: {r2_lstm:.2f}")
print(f"GRU RMSE: {rmse_gru:.2f}, R2: {r2_gru:.2f}")


# In[ ]:


# Hyper parameter


# In[68]:


#Hyperparameter optimisation
def build_lstm_model(hp):
    model = Sequential()
    # Tune the number of units in the first LSTM layer
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dropout(0.2))
    # Tune the number of units in the second LSTM layer
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(0.2))
    # Tune the number of units in the third LSTM layer
    model.add(LSTM(units=hp.Int('units_3', min_value=32, max_value=512, step=32)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mae')
    return model
tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=5,  # The number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to build and fit for each trial
    directory='my_dir',
    project_name='lstm_tuning'
)


# In[70]:


# Perform the hyperparameter search
tuner.search(X_train_reshaped, y_train, epochs=100, validation_data=(X_test_reshaped, y_test))
 
best_model = tuner.get_best_models(num_models=1)[0]


# In[72]:


# Remove old directory for GRU tuning
shutil.rmtree('my_dir/gru_tuning', ignore_errors=True)


# In[74]:


# Evaluate tuned GRU model
y_predicted_lstm = best_model.predict(X_test_reshaped)
y_predicted_lstm = scaler_y.inverse_transform(y_predicted_lstm)
y_predicted_lstm = pd.DataFrame(y_predicted_lstm, index=df.index[split_index:])


# In[78]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df['PX_LAST'][split_index:], lw=2, c='red')
plt.plot(df['PX_LAST'][split_index:].index, y_predicted_lstm, lw=2, c='blue')
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - LSTM Forecast')
plt.grid(True)
plt.show()


# In[80]:


shutil.rmtree('my_dir/lstm_tuning', ignore_errors=True)


# In[82]:


addMetrics(metricsDF, "LSTM prediction",y_predicted_lstm.values.reshape(-1))


# In[84]:


def build_gru_model(hp):
    model = Sequential()
    # Tune the number of units in the first GRU layer
    model.add(GRU(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='tanh', return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dropout(0.2))
    # Tune the number of units in the second GRU layer
    model.add(GRU(units=hp.Int('units_2', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(0.2))
    # Tune the number of units in the third GRU layer
    model.add(GRU(units=hp.Int('units_3', min_value=32, max_value=512, step=32)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mae')
    return model

tuner = kt.RandomSearch(
    build_gru_model,
    objective='val_loss',
    max_trials=5,  # The number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to build and fit for each trial
    directory='my_dir',
    project_name='gru_tuning'
)

# Perform the hyperparameter search
tuner.search(X_train_reshaped, y_train, epochs=100, validation_data=(X_test_reshaped, y_test))

best_model = tuner.get_best_models(num_models=1)[0]


# In[85]:


# Remove old directory for GRU tuning
shutil.rmtree('my_dir/gru_tuning', ignore_errors=True)


# In[92]:


# Evaluate tuned GRU model
y_predicted_gru = best_model.predict(X_test_reshaped)
y_predicted_gru = scaler_y.inverse_transform(y_predicted_gru)
y_predicted_gru = pd.DataFrame(y_predicted_gru, index=df.index[split_index:])


# In[94]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df['PX_LAST'][split_index:], lw=2, c='red')
plt.plot(df['PX_LAST'][split_index:].index, y_predicted_gru, lw=2, c='blue')
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - GRU Forecast')
plt.grid(True)
plt.show()


# In[96]:


addMetrics(metricsDF, "GRU prediction",y_predicted_gru.values.reshape(-1))
metricsDF


# In[ ]:


df1 = lstm_model

