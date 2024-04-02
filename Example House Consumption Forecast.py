# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:54:01 2023
House Power Consumption Forecast
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import requests
import certifi
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
import warnings
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import DateFormatter
import mysql.connector

import API_OpenMeteo

###############################################################################################################################
'Plot Parameters'
###############################################################################################################################
plt.rcParams['figure.figsize']= (20, 10)
plt.style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
date_format = DateFormatter('%m/%d %H:%M') 

###############################################################################################################################
'Functions'
###############################################################################################################################
'Function to define the season based on the month'
def define_season(month_number):
    if month_number in [12,1,2]:
        return 1
    elif month_number in [3,4,5]:
        return 2
    elif month_number in [6,7,8]:
        return 3
    elif month_number in [9,10,11]:
        return 4

'Function create_features'
def create_features(df):
    """
    Creates date/time features from a dataframe 
    
    Args:
        df - dataframe with a datetime index
        
    Returns:
        df - dataframe with 'Weekofyear','Dayofyear','Month','Dayofmonth',
             'Dayofweek','Weekend','Season','Holiday','Hour' and 'Minute' features created
    """
    
    df['Date'] = df.index
    df['Weekofyear'] = df['Date'].dt.weekofyear   #Value: 1-52
    df['Dayofyear'] = df['Date'].dt.dayofyear    #Value: 1-365
    df['Dayofmonth'] = df['Date'].dt.day   #Value: 1-30/31
    df['Dayofweek']= df['Date'].dt.weekday+1     #Value: 1-7 (Monday-Sunday)
    df['Weekend']= np.where((df['Dayofweek']==6) | (df['Dayofweek']==7), 1, 0)    #Value: 1 if weekend, 0 if not 
    df['Hour'] = df['Date'].dt.hour
    df['Hour']= (df['Hour']+24).where(df['Hour']==0, df['Hour'])    #Value: 1-24
    df['Minute']= df['Date'].dt.minute     #Value: 0, 15, 30 or 45
    df= df.drop(['Date'], axis=1)
    
    return df

'Function lag_features'
def lag_features(lag_dataset, days_list, var):
    
    temp_data = lag_dataset[var]
    
    for days in days_list:
        rows = 96 * days
        lag_dataset[var + "_lag_{}".format(days)] = temp_data.shift(rows)

    return lag_dataset 

'Function cyclical_features'
def cyclical_features(df):
    """
    Transforms (date/time) features into cyclical sine and cosine features
    
    Args:
        df - dataframe with 'Weekofyear','Dayofyear','Season','Month',
             'Dayofmonth','Dayofweek','Hour','Minute' columns
        
    Returns:
        df - dataframe including the cyclical features (x and y for each column)
    """
    
    df['Weekofyear_x']= np.cos(df['Weekofyear']*2*np.pi/52)
    df['Weekofyear_y']= np.sin(df['Weekofyear']*2*np.pi/52)
    df['Dayofyear_x']= np.cos(df['Dayofyear']*2*np.pi/365)
    df['Dayofyear_y']= np.sin(df['Dayofyear']*2*np.pi/365)
    df['Dayofmonth_x']= np.cos(df['Dayofmonth']*2*np.pi/31)
    df['Dayofmonth_y']= np.sin(df['Dayofmonth']*2*np.pi/31)
    df['Dayofweek_x']= np.cos(df['Dayofweek']*2*np.pi/7)
    df['Dayofweek_y']= np.sin(df['Dayofweek']*2*np.pi/7)
    df['Hour_x']= np.cos(df['Hour']*2*np.pi/24)
    df['Hour_y']= np.sin(df['Hour']*2*np.pi/24)
    df['Minute_x']= np.cos(df['Minute']*2*np.pi/45)
    df['Minute_y']= np.sin(df['Minute']*2*np.pi/45)
    df= df.drop(columns=['Weekofyear','Dayofyear', 'Dayofmonth',
                                         'Dayofweek','Hour','Minute'])
    
    return df


'##########################################Load (House Power Consumption) Forecast################################################'

###############################################################################################################################
'Getting the weather forecasts for next_hours from OpenWeather API'
###############################################################################################################################
df_forecast = API_OpenMeteo.weather_forecast_15min
start_forecast = df_forecast.index[0]
end_forecast = df_forecast.index[-1]

###############################################################################################################################
'Getting power consumption data from the database'
print('###################################Power consumption data######################################')
###############################################################################################################################
# Define your database connection parameters
host = 'db.tecnico.ulisboa.pt'
database = 'ist1100758'
user = 'ist1100758'
password = 'ckpx5936'

# Establish a database connection
try:
    connection = mysql.connector.connect(
        host = host,
        database = database,
        user = user,
        password = password
    )
    
    if connection.is_connected():
        print("Connected to the database")

    # SQL query to retrieve the data from the 'Weather' table
    query = f"SELECT * FROM ist1100758.Power_cons WHERE Date < '{end_forecast}'"

    # Creating a dataframe with the data 
    historical_cons = pd.read_sql(query, connection)
    historical_cons.set_index('Date', inplace= True)
    print("Power consumption data successfully loaded from the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        connection.close()
        print("Database connection closed")

###############################################################################################################################
'Getting historical weather data from the database'
print('###################################Historical weather data#####################################')
###############################################################################################################################
# Define your database connection parameters
host = 'db.tecnico.ulisboa.pt'
database = 'ist1100758'
user = 'ist1100758'
password = 'ckpx5936'

# Establish a database connection
try:
    connection = mysql.connector.connect(
        host = host,
        database = database,
        user = user,
        password = password
    )
    
    if connection.is_connected():
        print("Connected to the database")

    # SQL query to retrieve the data from the 'Weather' table
    query = f"SELECT * FROM Demo_Weather WHERE Date < '{start_forecast}'"

    # Creating a dataframe with the data 
    historical_weather = pd.read_sql(query, connection)
    historical_weather.set_index('Date', inplace= True)
    print("Historical weather data successfully loaded from the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        connection.close()
        print("Database connection closed")

###############################################################################################################################
'Creating final dataframe'
print('##########################################Forecasting##########################################')
###############################################################################################################################
# Merging power consumption data with weather data
df_train = pd.merge(historical_cons, historical_weather, left_index=True, right_index=True)
df_test = pd.merge(historical_cons, df_forecast, left_index=True, right_index=True)

# Creating final dataframe (training + forecast)
df_final = df_train.append(df_test)

# Creating date/time features using datetime column Date as index
df_final = create_features(df_final)

# Creating lag features of power consumption for 2,3,4 and 5 days before'
df_final = lag_features(df_final,[2,3,4,5], 'house_consumption')
df_final.dropna(inplace=True, subset= df_final.columns[1:])

# Transforming date/time features into two dimensional features'
df_final = cyclical_features(df_final)

###############################################################################################################################
'Feature selection'
###############################################################################################################################
# Correlation matrix
corr = df_final.corr()[['house_consumption']].sort_values(by= 'house_consumption', ascending= False).round(2)

# Heatmap feature correlation
fig = plt.subplots(figsize=(12, 10))
heatmap = sns.heatmap(corr[1:], vmin= -1, vmax= 1, annot= True, cmap= 'BrBG')
heatmap.set_title('Features correlation with power consumption', alpha= 0.75, weight= "bold", pad= 10)
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 16)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 16)
plt.show()

# Array containing the names of all features available
all_features = df_final.columns.values.tolist()
all_features.remove('house_consumption')
all_features= np.array(all_features) 

X = df_final.values
Y = X[:,0] 
X = X[:,[x for x in range(1,len(all_features)+1)]]

# Feature importance for the model
parameters_XGBOOST = {'n_estimators' : 500,
                  'learning_rate' : 0.01,
                  'verbosity' : 0,
                  'n_jobs' : -1,
                  'gamma' : 0,
                  'min_child_weight' : 1,
                  'max_delta_step' : 0,
                  'subsample' : 0.7,
                  'colsample_bytree' : 1,
                  'colsample_bylevel' : 1,
                  'colsample_bynode' : 1,
                  'reg_alpha' : 0,
                  'reg_lambda' : 1,
                  'random_state' : 18,
                  'objective' : 'reg:linear',
                  'booster' : 'gbtree'}

reg_XGBOOST = xgb.XGBRegressor(**parameters_XGBOOST)
reg_XGBOOST.fit(X, Y)
importance = pd.DataFrame(data= {'Feature': all_features, 'Score': reg_XGBOOST.feature_importances_})
importance = importance.sort_values(by= ['Score'], ascending= False)
importance.set_index('Feature', inplace= True)

###############################################################################################################################
'Forecasting'
###############################################################################################################################
# Defining the number of features to use
num_features = 15  # Optimal number of features is 15 (total: 22 features available)

# Defining training and test periods
data_train = df_final.loc[: df_train.index[-1]]
data_test = df_final.loc[df_forecast.index[0] :]

# Plot train-test
fig,ax = plt.subplots()
coloring = df_final['house_consumption'].max()
plt.plot(data_train.index, data_train['house_consumption'], color= "darkcyan", alpha= 0.75)
plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
plt.plot(data_test.index, data_test['house_consumption'], color = "dodgerblue", alpha= 0.60)
plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75,weight= "bold", rotation= 45)
plt.yticks(alpha= 0.75,weight= "bold")
plt.title(" Train - Test split for power consumption", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Features used
USE_COLUMNS = importance[:num_features].index.values

# Forecasting variable
FORECAST_COLUMN = ['house_consumption']

print('The features used in the XGBOOST model are:', USE_COLUMNS)

# XGBOOST model
xtrain = data_train.loc[:, USE_COLUMNS]
xtest = data_test.loc[:, USE_COLUMNS]
ytrain = data_train.loc[:, FORECAST_COLUMN]
ytest = data_test.loc[:, FORECAST_COLUMN]

reg_XGBOOST.fit(xtrain, np.ravel(ytrain))

# Predictions and post-processing
df_XGBOOST = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
df_XGBOOST['Prediction']= np.where(df_XGBOOST['Prediction']< 0, 0 , df_XGBOOST['Prediction'])
df_XGBOOST['Real'] = ytest

# Regression plot
sns.scatterplot(data= df_XGBOOST, x='Real', y= 'Prediction')
plt.plot(ytest, ytest, color = "red", linewidth= 1) 
plt.xlabel("Real Energy (kWh)", alpha= 0.75, weight= "bold")
plt.ylabel("Predicted Energy (kWh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold")
plt.yticks(alpha= 0.75, weight= "bold")
plt.title("Correlation real vs predictions for power consumption", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Real vs predictions in the same plot
fig, ax = plt.subplots()
sns.lineplot(x= df_XGBOOST.index, y= df_XGBOOST.Real, label= "Real", ax= ax, linestyle = "dashed")
sns.lineplot(x= df_XGBOOST.index, y= df_XGBOOST.Prediction, label= "Predicted", ax= ax, linewidth = 0.5)
plt.gca().xaxis.set_major_formatter(date_format)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Energy (kWh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", rotation= 45)
plt.yticks(alpha= 0.75, weight= "bold")
plt.legend(loc= 'upper left')
plt.title("Real vs predicted power consumption", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Errors
MAE_XGBOOST = metrics.mean_absolute_error(df_XGBOOST.Real, df_XGBOOST.Prediction)
RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_XGBOOST.Real, df_XGBOOST.Prediction))
normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest['house_consumption'].max()
R2_XGBOOST = metrics.r2_score(df_XGBOOST.Real, df_XGBOOST.Prediction)

print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
print('XGBOOST - R square (%):', round(R2_XGBOOST,2))

###############################################################################################################################
'Forecast results'
###############################################################################################################################
# Predictions
predictions_load = df_XGBOOST[['Prediction']]
predictions_load.rename(columns= {'Prediction':'house_consumption'}, inplace= True)