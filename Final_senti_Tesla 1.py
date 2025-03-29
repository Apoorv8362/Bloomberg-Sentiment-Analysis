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


# In[6]:


# User input for equity and dates
tickers = input("Enter the Equity name. eg: AAPL US Equity\n")
commands = "PX_OPEN, PX_LAST, PX_HIGH, PX_LOW, PX_VOLUME, PX_CLOSE,TWITTER_SENTIMENT, TWITTER_NEG_SENTIMENT_COUNT, TWITTER_POS_SENTIMENT_COUNT, TWITTER_NEUTRAL_SENTIMENT_CNT, TWITTER_PUBLICATION_COUNT,NEWS_POS_SENTIMENT_COUNT, NEWS_NEG_SENTIMENT_COUNT, NEWS_NEUTRAL_SENTIMENT_COUNT, NEWS_PUBLICATION_COUNT"


# In[7]:


start = input("Enter the start date in YYYY-MM-DD format\n")
a = input("Type YES if you want current date as end date or no for custom date\n")
#2015-01-02
#2024-05-31

if a.upper() == "YES":
    end = str(date.today())
else:
    end = str(input("Enter the end date in YYYY-MM-DD format\n"))

# Create a unique filename based on inputs
filename = hashlib.md5(''.join((tickers, "+", commands, "+", start, "+", end)).encode('utf-8')).hexdigest()

# Load or fetch data
if os.path.exists(filename + '.csv'):
    data = pd.read_csv(filename + ".csv", header=[0, 1], parse_dates=True, index_col=0)
else:
    data = blp.bdh(tickers=tickers.split(', '), flds=commands.split(', '), start_date=start, end_date=end, Per='D', Fill='P', Days='A', adjust='all')
    data.to_csv(filename + ".csv")

df = data[tickers]


# In[8]:


df.info()


# In[9]:


df.isna()


# In[10]:


#df.fillna(df.median(), inplace=True)
df.dropna()


# In[11]:


# Define technical indicators

def get_technical_indicators(dataset):
    dataset['ma7'] = dataset['PX_LAST'].rolling(window=7).mean()
    dataset['ma21'] = dataset['PX_LAST'].rolling(window=21).mean()
    dataset['26ema'] = dataset['PX_LAST'].ewm(span=26).mean()
    dataset['12ema'] = dataset['PX_LAST'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    dataset['20sd'] = dataset['PX_LAST'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    dataset['ema'] = dataset['PX_LAST'].ewm(com=0.5).mean()
    dataset['momentum'] = dataset['PX_LAST'] - 1
    return dataset

ti_df = get_technical_indicators(df[['PX_LAST']].copy()).fillna(method='bfill')
ti_df.index = pd.DatetimeIndex(df.index)


ti_df


# In[12]:


ti_df = ti_df.drop(columns=['PX_LAST'])


# In[13]:


ti_df


# In[14]:


target_column = ['PX_LAST']


# In[15]:


df1 = df.join(ti_df)


# In[16]:


df1


# In[17]:


# Prepare data for modeling

predictors = list(set(df.columns) - set(target_column))

# Scale data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X = scaler_X.fit_transform(df[predictors])
y = scaler_y.fit_transform(df[target_column].values.reshape(-1, 1))



# In[18]:


# Split data
split_index = int(len(df) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape for LSTM and GRU
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))



# In[19]:


X_train.shape


# In[20]:


X_test.shape


# In[21]:


X_train


# In[22]:


train_dates = df.index[:split_index]
test_dates = df.index[split_index:]

# Plotting the training data
plt.figure(figsize=(14, 7))
plt.plot(train_dates, y_train, color='green', label='Training Data')

# Plotting the testing data
plt.plot(test_dates, y_test, color='red', label='Testing Data')

# Adding titles and labels
plt.title('Training and Testing Data')
plt.xlabel('Date')
plt.ylabel('Target Variable')

# Showing the legend
plt.legend()

# Display the plot
plt.show()


# In[23]:


print(X_train.shape[2])


# In[24]:


# Define the LSTM model
def create_lstm(X_train, regress=False):
    model = Sequential()
    model.add(LSTM(units=32, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


"""#Testing new model 
# Build model - LSTM with 50 neurons and 4 hidden layers  
def create_lstm(X_train, regress=False):
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1), activation='tanh'))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, activation='tanh'))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, activation='tanh'))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, activation='tanh'))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 200, batch_size = 64)"""


# In[25]:


modelName = "./LSTM"
if not os.path.exists(modelName + "/model.h5"):
    model_lstm = create_lstm(X_train, regress=False)
    model_lstm.compile(loss='mae', optimizer='adam')
    history_lstm = model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), shuffle=False)
    scores_lstm = model_lstm.evaluate(X_train, y_train, verbose=0)
    model_lstm.save(modelName + ".weights.h5")
    model_lstm.save_weights(modelName + "model.weights.h5")
else:
    model_lstm = load_model(modelName+ ".weights.h5")
    model_lstm.load_weights(modelName + 'model.weights.h5')
    model_lstm.compile(loss='mae', optimizer='adam')
    scores_lstm = model_lstm.evaluate(X_train, y_train, verbose=0)
 
print(model_lstm.summary())
print('')


# In[26]:


# Define the GRU model
def create_gru(X_train, regress=False):
    model = Sequential()
    model.add(GRU(units=32, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=512))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model


# In[27]:


modelName = "./GRU"
if not os.path.exists(modelName+"/model.h5"):
    model_gru = create_gru(X_train, regress=False)
    model_gru.compile(loss = 'mae', optimizer = 'adam')
    history_gru = model_gru.fit(X_train, y_train, epochs = 15,batch_size=32, validation_data = (X_test, y_test), shuffle=False)
    scores_gru = model_gru.evaluate(X_train, y_train, verbose=0)
    model_gru.save(modelName + ".weights.h5")
    # serialize weights to HDF5
    model_gru.save_weights(modelName+"model.weights.h5")
else:    
    model_gru = load_model(modelName + ".weights.h5")
    model_gru.load_weights(modelName+'model.weights.h5')
    results_gru = model_gru.compile(loss = 'mae', optimizer = 'adam')
    history_gru = model_gru.fit(X_train, y_train, epochs = 15,batch_size=32, validation_data = (X_test, y_test), shuffle=False)
    scores_gru= model_gru.evaluate(X_train, y_train, verbose=0)

print(model_gru.summary())
print('')


# In[28]:


# Metrics evaluation
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


# In[29]:


# Make predictions and plot results
y_predicted_lstm = model_lstm.predict(X_test)
y_predicted_lstm = scaler_y.inverse_transform(y_predicted_lstm)
y_predicted_lstm = pd.DataFrame(y_predicted_lstm)

plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df.index[split_index:], df['PX_LAST'][split_index:], lw=2, c='blue')
plt.plot(df.index[split_index:], y_predicted_lstm, lw=2, c='red')
plt.legend(['Series', 'LSTM Forecast'])
plt.title(tickers + ' - LSTM Forecast')
plt.grid(True)
plt.show()

addMetrics(metricsDF, "LSTM prediction", y_predicted_lstm.values.reshape(-1))

y_predicted_gru = model_gru.predict(X_test)
y_predicted_gru = scaler_y.inverse_transform(y_predicted_gru)
y_predicted_gru = pd.DataFrame(y_predicted_gru)

plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df.index[split_index:], df['PX_LAST'][split_index:], lw=2, c='blue')
plt.plot(df.index[split_index:], y_predicted_gru, lw=2, c='green')
plt.legend(['Series', 'GRU Forecast'])
plt.title(tickers + ' - GRU Forecast')
plt.grid(True)
plt.show()

addMetrics(metricsDF, "GRU prediction", y_predicted_gru.values.reshape(-1))

# Show metrics
print(metricsDF)


# In[30]:


# Remove old directory if it exists
shutil.rmtree('my_dir/lstm_tuning', ignore_errors=True)


# In[31]:


dfname=tickers


# In[32]:


#Hyperparameter optimisation
def build_lstm_model(hp):
    model = Sequential()
    # Tune the number of units in the first LSTM layer
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
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


# In[33]:


# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
 
best_model = tuner.get_best_models(num_models=1)[0]


# In[34]:


# Remove old directory for GRU tuning
shutil.rmtree('my_dir/gru_tuning', ignore_errors=True)


# In[35]:


# Evaluate tuned GRU model
y_predicted_lstm = best_model.predict(X_test)
y_predicted_lstm = scaler_y.inverse_transform(y_predicted_lstm)
y_predicted_lstm = pd.DataFrame(y_predicted_lstm, index=df.index[split_index:])


# In[36]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df['PX_LAST'][split_index:], lw=2, c='red')
plt.plot(df['PX_LAST'][split_index:].index, y_predicted_lstm, lw=2, c='blue')
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - LSTM Forecast')
plt.grid(True)
plt.show()


# In[37]:


shutil.rmtree('my_dir/lstm_tuning', ignore_errors=True)


# In[38]:


addMetrics(metricsDF, "LSTM prediction",y_predicted_lstm.values.reshape(-1))


# In[39]:


def build_gru_model(hp):
    model = Sequential()
    # Tune the number of units in the first GRU layer
    model.add(GRU(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
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
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

best_model = tuner.get_best_models(num_models=1)[0]


# In[40]:


# Remove old directory for GRU tuning
shutil.rmtree('my_dir/gru_tuning', ignore_errors=True)


# In[41]:


# Evaluate tuned GRU model
y_predicted_gru = best_model.predict(X_test)
y_predicted_gru = scaler_y.inverse_transform(y_predicted_gru)
y_predicted_gru = pd.DataFrame(y_predicted_gru, index=df.index[split_index:])


# In[42]:


plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.plot(df['PX_LAST'][split_index:], lw=2, c='red')
plt.plot(df['PX_LAST'][split_index:].index, y_predicted_gru, lw=2, c='blue')
plt.legend(['Series', 'Forecast'])
plt.title(dfname+' - GRU Forecast')
plt.grid(True)
plt.show()


# In[43]:


addMetrics(metricsDF, "GRU prediction",y_predicted_gru.values.reshape(-1))
metricsDF


# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


get_ipython().system('pip install yfinance')


# In[45]:


get_ipython().system('pip install xgboost')


# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

get_ipython().run_line_magic('matplotlib', 'inline')

# Import the models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# In[47]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[48]:


print(X_train)
print(y_train)


# In[49]:


df


# In[50]:


df.fillna(df.median(), inplace=True)


# In[51]:


df_adj = df[['PX_LAST']]
df_adj 


# In[52]:


forecast_out = 30

# Create column for target variable shifted 'n' days up
df_adj['Prediction'] = df_adj[['PX_LAST']].shift(-forecast_out)

df_adj


# In[53]:


X = np.array(df_adj.drop(['Prediction'],axis = 1))
# Remove last 'n' rows
X = X[:-forecast_out]

print(X)


# In[54]:


y = np.array(df_adj['Prediction'])
# Remove last 'n' rows
y = y[:-forecast_out]

print(y)


# In[55]:


train_size = int(X.shape[0]*0.7)

X_train = X[0:train_size]
y_train = y[0:train_size]

X_test = X[train_size:]
y_test = y[train_size:]


# In[56]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[57]:


print(X_train)
print(y_train)


# In[58]:


X_forecast = np.array(df_adj.drop(['Prediction'],axis = 1))[-forecast_out:]
print(X_forecast)


# In[60]:


models = {}

models["Linear"] = LinearRegression()
models["Lasso"] = Lasso()
models["Ridge"] = Ridge()
models["ElasticNet"] = ElasticNet()
models["KNN"] = KNeighborsRegressor()
models["DecisionTree"] = DecisionTreeRegressor()
models["SVR"] = SVR(kernel='rbf', C=1e3, gamma='scale')
models["AdaBoost"] = AdaBoostRegressor()
models["GradientBoost"] = GradientBoostingRegressor()
models["RandomForest"] = RandomForestRegressor()
models["ExtraTrees"]= ExtraTreesRegressor()
models["BaggingRegressor"] = BaggingRegressor()
models["XGBRegressor"] = XGBRegressor(objective ='reg:squarederror')
models["MLPRegressor"] = MLPRegressor(solver = 'lbfgs')
     


# In[61]:


model_results = []  
model_names   = []
for model_name in models:
    model = models[model_name]
    # TimeSeries Cross validation
    tscv = TimeSeriesSplit(n_splits=7)
    
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    model_results.append(cv_results)
    model_names.append(model_name)
    print("{}: {}, ({})".format(model_name, round(cv_results.mean(), 6), round(cv_results.std(), 6)))
     


# In[62]:


# Compare algorithms  

def box_compare():
  sns.set(rc={'figure.figsize':(15, 9)})
  sns.set_style(style='whitegrid', )
  figure = plt.figure()
  plt.title('Regression models comparison', color = 'black', fontsize = 20)
  axis = figure.add_subplot(111)
  plt.boxplot(model_results)
  axis.set_xticklabels(model_names, rotation = 45, ha="right")
  axis.set_ylabel("R^2 score")
  plt.margins(0.05, 0.1)

box_compare()
     


# In[63]:


# Create Linear Regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)# Create Linear Regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)


# In[64]:


# Make predictions using the model
predictions = lr.predict(X_test)


# In[65]:


# The coefficients
print('Coefficients: ', lr.coef_)
# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(y_test, predictions, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(y_test, predictions), 3)))


# In[66]:


# Plot predictions against actual Adjusted Close prices 

x_axis = np.array(range(0, predictions.shape[0]))
plt.plot(x_axis, y_test, color='g', label="actual")
plt.plot(x_axis, predictions, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('Linear Regression - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()


# In[67]:


pd.DataFrame({"Actual": y_test, "Predict": predictions}).head()


# In[68]:


# create basic scatterplot
plt.scatter(y_test,predictions)

# obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(y_test, predictions, 1)

# add linear regression line to scatterplot 
plt.plot(y_test, m*y_test+b, c='r')

plt.xlabel("Prices: ")
plt.ylabel("Predicted prices: ")
plt.title("Prices vs Predicted prices:  vs ")
plt.show()
     


# In[69]:


# Predicted prices

lr_prediction = lr.predict(X_forecast)
print(lr_prediction)


# In[70]:


# Actual prices

X_forecast


# In[71]:


# The coefficients
print('Coefficients: ', lr.coef_)
# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(X_forecast, lr_prediction, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(X_forecast, lr_prediction), 3)))


# In[72]:


# Plot predictions against actual Adjusted Close prices 

x_axis = np.array(range(0, lr_prediction.shape[0]))
plt.plot(x_axis, X_forecast, color='g', label="actual")
plt.plot(x_axis, lr_prediction, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('Linear Regression - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()


# In[73]:


# Reshape data to be 1D
X_forecast = X_forecast.reshape(-1)
X_forecast 


# In[74]:


pd.DataFrame({"Actual": X_forecast, "Predict": lr_prediction}).head()


# In[75]:


# Hyper parameter optimisation


# In[76]:


# Create dictionary of parameters
parameters = { 'fit_intercept': [True, False],
         'n_jobs': [None, -1]}


# In[77]:


# Grid search to find best parameters
gridsearchcv_lr = GridSearchCV(estimator=lr, param_grid=parameters, cv=tscv, scoring='r2')
grid_result_lr = gridsearchcv_lr.fit(X_train, y_train)
print("Best: {} using {}".format(grid_result_lr.best_score_, grid_result_lr.best_params_))


# In[78]:


# All of the best parameters for the optimal model

best_model_lr = grid_result_lr.best_estimator_
print(f"Best model has the following hyperparameters: {best_model_lr}")


# In[79]:


# Reshape data to be 2D
X_forecast = X_forecast.reshape(-1,1)
X_forecast


# In[80]:


# Create Linear Regression model
lr = LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None)

# Train the model using the training sets
lr.fit(X_train, y_train)

# Make predictions using the model
lr_predictions = lr.predict(X_forecast)
predictions = lr.predict(X_test)


# The coefficients
print('Coefficients: ', lr.coef_)
# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(y_test, predictions, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(y_test, predictions), 3)))



# The coefficients
print('Coefficients: ', lr.coef_)
# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(X_forecast, lr_predictions, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(X_forecast, lr_predictions), 3)))


# In[81]:


# Plot predictions against actual Adjusted Close prices  

x_axis = np.array(range(0, lr_predictions.shape[0]))
plt.plot(x_axis, X_forecast, color='g', label="actual")
plt.plot(x_axis, lr_predictions, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('Linear Regression - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()


# In[82]:


# MLP regressor


# In[83]:


# Create an MLP Regressor model  
mlpr = MLPRegressor(max_iter=500, solver = 'lbfgs')

# Train the model using the training sets
mlpr.fit(X_train, y_train)
mlpr


# In[84]:


# Make predictions using the model
predictions = mlpr.predict(X_test)


# In[85]:


# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(y_test, predictions, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(y_test, predictions), 3)))


# In[86]:


# Plot predictions against actual Adjusted Close prices 

x_axis = np.array(range(0, predictions.shape[0]))
plt.plot(x_axis, y_test, color='g', label="actual")
plt.plot(x_axis, predictions, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('MLP Regressor - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()


# In[88]:


pd.DataFrame({"Actual": y_test, "Predict": predictions}).head()


# In[89]:


# create basic scatterplot
plt.scatter(y_test,predictions)

# obtain m (slope) and b(intercept) of linear regression line
m, b = np.polyfit(y_test, predictions, 1)

# add linear regression line to scatterplot 
plt.plot(y_test, m*y_test+b, c='r')

plt.xlabel("Prices: ")
plt.ylabel("Predicted prices: ")
plt.title("Prices vs Predicted prices:  vs ")
plt.show()


# In[90]:


# Predicted prices

mlpr_prediction = mlpr.predict(X_forecast)
print(mlpr_prediction)


# In[91]:


X_forecast


# In[92]:


# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(X_forecast, mlpr_prediction, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(X_forecast, mlpr_prediction), 3)))


# In[93]:


# Plot predictions against actual Adjusted Close prices 

x_axis = np.array(range(0, mlpr_prediction.shape[0]))
plt.plot(x_axis, X_forecast, color='g', label="actual")
plt.plot(x_axis, mlpr_prediction, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('MLP Regressor - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()


# In[94]:


# Reshape data to be 1D
X_forecast = X_forecast.reshape(-1)


# In[95]:


pd.DataFrame({"Actual": X_forecast, "Predict": mlpr_prediction}).head()


# In[96]:


# Reshape data as this needs to be 2D
y_train = y_train.reshape(-1, 1)


# In[97]:


# Normalise data to improve convergence

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_norm = scaler.fit_transform(X_train)
y_train_norm = scaler.transform(y_train)
X_test_norm = scaler.transform(X_test)


# In[98]:


# Create model

mlpr = MLPRegressor(max_iter=500)


# In[99]:


# Create dictionary of parameters to iterate over

parameters = {"hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
              "activation": ["identity", "logistic", "tanh", "relu"], 
              "solver": ["lbfgs", "sgd", "adam"], 
              "alpha": [0.0001, 0.05], 
              "learning_rate": ['constant','adaptive']}


# In[ ]:


# Grid search to find best parameters

gridsearchcv_mlpr = GridSearchCV(estimator=mlpr, param_grid=parameters, cv=tscv, scoring='r2')
grid_result_mlpr = gridsearchcv_mlpr.fit(X_train_norm, y_train_norm)
print("Best: {} using {}".format(grid_result_mlpr.best_score_, grid_result_mlpr.best_params_))


# In[ ]:


# All of the best parameters for the optimal model  

best_model_mlpr = grid_result_mlpr.best_estimator_
print(f"Best model has the following hyperparameters: {best_model_mlpr}")


# In[ ]:


# Reshape data to be 2D
X_forecast = X_forecast.reshape(-1,1)
X_forecast


# In[ ]:


# Create MLP Regression model  
mlpr = MLPRegressor(activation='relu', alpha=0.05, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(100,), learning_rate='constant',
             learning_rate_init=0.001, max_fun=15000, max_iter=500,
             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
             power_t=0.5, random_state=None, shuffle=True, solver='lbfgs',
             tol=0.0001, validation_fraction=0.1, verbose=False,
             warm_start=False)

# Train the model using the training sets
mlpr.fit(X_train, y_train)

# Make predictions using the model
mlpr_predictions = mlpr.predict(X_forecast)
predictions = mlpr.predict(X_test)


# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(y_test, predictions, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(y_test, predictions), 3)))


# The mean squared error
print('RMSE: {}'.format(round(mean_squared_error(X_forecast, mlpr_prediction, squared=False), 3)))
# Explained variance score: 1 is perfect prediction, 0 is random
print('R^2 score: {}'.format(round(r2_score(X_forecast, mlpr_prediction), 3)))
     


# In[ ]:


# Plot predictions against actual Adjusted Close prices

x_axis = np.array(range(0, mlpr_predictions.shape[0]))
plt.plot(x_axis, X_forecast, color='g', label="actual")
plt.plot(x_axis, mlpr_predictions, color='r', label="predictions")
plt.xlabel('Time periods')
plt.ylabel('Adjusted Close price')
plt.title('MLP Regressor - Predictions vs Actual Prices')
plt.legend(loc='lower right')
plt.show()
     


# In[ ]:





# In[ ]:





# In[ ]:




