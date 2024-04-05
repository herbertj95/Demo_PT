# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:27:22 2023
House EV Connection and Req Forecast
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import itertools

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

'Function plot_confusion_matrix'
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


'##########################################House EV Connection and Req Forecast###############################################'

# Define the time horizon of the forecast (in hours)
horizon_forecast = 36

# Define the number of days of historical weather data to get (before the start of the forecast)
days_past_weather = 7

# Define the latitude and longitude of the location
lat = 38.954341
lon = -8.9873593

###############################################################################################################################
'Getting the weather forecasts for next_hours from OpenWeather API'
###############################################################################################################################
df_forecast = API_OpenMeteo.get_weather_data(lat, lon, horizon_forecast, days_past_weather)
start_forecast = df_forecast.index[0]
end_forecast = df_forecast.index[-1]

###############################################################################################################################
'Getting EV data from the database'
print('###################################EV connection and req data######################################')
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

    # SQL query to retrieve the data from the 'EV_house' table
    query = f"SELECT * FROM ist1100758.EV_house WHERE Date < '{end_forecast}'"

    # Creating a dataframe with the data 
    EV_data = pd.read_sql(query, connection)
    EV_data.set_index('Date', inplace= True)
    print("EV data successfully loaded from the database")

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
df_train = pd.merge(EV_data, historical_weather, left_index=True, right_index=True)
df_test = pd.merge(EV_data, df_forecast, left_index=True, right_index=True)

# Creating final dataframe (training + forecast)
data_final = df_train.append(df_test)

variables = ['ev_connection', 'ev_req']
# Distributing energy consumed value equally during the period the EV is charging
count = 0
energy = None
first_zero = True

for i, value in enumerate(data_final[variables[0]]):
    # Cheking if the 'Connection' is 1, if it is, count the number of ones and distribute the 'Energy req' value among them
    if value == 1:
        count += 1
        if data_final[variables[1]][i] != 0:
            energy = data_final[variables[1]][i]
            replacement = energy / count
            data_final.loc[i-count+1:i+1, variables[1]] = replacement
            count = 0
            energy = None
        
# Looking for the errors in the distribution of the energy
error = data_final[(data_final[variables[0]] == 0) & (data_final[variables[1]] != 0) | (data_final[variables[0]] == 1) & (data_final[variables[1]] == 0)]

# Creating date/time features using datetime column Date as index
data_final = create_features(data_final)

# Barplot average energy consumption per hour
# mean_per_hour = data_final.groupby('Hour')[variables[1]].agg(["mean"])
# fig, ax = plt.subplots()
# ax.plot(mean_per_hour.index, mean_per_hour["mean"], color= 'darkcyan')
# plt.xticks(range(1,25), alpha= 0.75, weight= "bold")
# plt.yticks(alpha=0.75, weight= "bold")
# plt.xlabel("Hour", alpha=0.75, weight= "bold")
# plt.ylabel("Energy (Wh)", alpha= 0.75, weight= "bold")
# plt.title("Average energy consumption per hour", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
# plt.show()

# Creating lag features of power consumption for 2, 3, 4 and 5 days
data_final = lag_features(data_final,[2,3,4,5], variables[1])
data_final.dropna(inplace=True, subset= data_final.columns[2:])

# Transforming date/time features into two dimensional features
data_final = cyclical_features(data_final)

#Defining training and test periods
data_train = data_final.loc[: start_forecast - timedelta(minutes= 15)]
data_test = data_final.loc[start_forecast :]

###############################################################################################################################
'Forecasting Connection (Classification)'
print('Forecast variable: ', variables[0])
###############################################################################################################################
#Plot train-test
fig,ax = plt.subplots()
coloring = data_final[variables[0]].max()
plt.plot(data_train.index, data_train[variables[0]], color= "darkcyan", alpha= 0.75)
plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
plt.plot(data_test.index, data_test[variables[0]], color = "dodgerblue", alpha= 0.60)
plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Plugged (1), Unplugged (0)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
plt.title("Train - Test split for "+ variables[0], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()
 
# Array containing the names of all features available (removing variables)
all_features = data_final.columns.values.tolist()
all_features.remove(variables[0])
all_features.remove(variables[1])
all_features= np.array(all_features) 
    
X = data_final.values
Y = X[:, 0] 
X = X[:,[x for x in range(2,len(all_features)+2)]]

# Feature importance for the model
cla_XGBOOST = xgb.XGBClassifier()
cla_XGBOOST.fit(X,Y)

importance = pd.DataFrame(data= {'Feature': all_features, 'Score': cla_XGBOOST.feature_importances_})
importance = importance.sort_values(by= ['Score'], ascending= False)
importance.set_index('Feature', inplace= True)

# Defining the number of features to use in the models'
num_features = 20       #Optimal number of features is 20  

# Features used
USE_COLUMNS = importance[:num_features].index.values
print('The features used in the XGBOOST model are:', USE_COLUMNS)

# Forecasting variable
FORECAST_COLUMN = [variables[0]]

# XGBOOST model
xtrain = data_train.loc[:, USE_COLUMNS]
xtest = data_test.loc[:, USE_COLUMNS]
ytrain = data_train.loc[:, FORECAST_COLUMN]
ytest = data_test.loc[:, FORECAST_COLUMN]

cla_XGBOOST.fit(xtrain, ytrain)

# Predictions
df_cla = pd.DataFrame(cla_XGBOOST.predict(xtest), columns=['Prediction'], index= xtest.index)
df_cla['Real']= ytest

# Accuracy
accuracy = accuracy_score(df_cla.Real, df_cla.Prediction)
print(f'XGBOOST Classifier Accuracy for Plug: {accuracy:.2f}')

# Confusion matrix
cnf_matrix = confusion_matrix(df_cla.Real, df_cla.Prediction, labels=[1,0])
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes= ['Plugged= 1','Plugged= 0'], normalize= False, title= 'Confusion matrix for '+ variables[0])
plt.show()

# Real vs predictions plot
fig,ax = plt.subplots()
ax.plot(df_cla.Real, label= "Real")
ax.plot(df_cla.Prediction, label= "Predicted", ls= '--')
plt.xlabel("Date", alpha= 0.75, weight="bold")
plt.ylabel('Plugged (1), Unplugged (0)', alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold",fontsize= 11)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.legend(frameon= False, loc= 'best')
plt.title("Correlation real vs predicted for "+ variables[0], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

###############################################################################################################################
'Forecasting Energy Req (Regression)'
print('#################################################################')
print('Forecast variable: ', variables[1])
###############################################################################################################################
# Plot train-test
fig,ax = plt.subplots()
coloring = data_final[variables[1]].max()
plt.plot(data_train.index, data_train[variables[1]], color= "darkcyan", alpha= 0.75)
plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
plt.plot(data_test.index, data_test[variables[1]], color = "dodgerblue", alpha= 0.60)
plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Energy (Wh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11)
plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
plt.title("Train - Test split for " + variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

#Array containing the names of all features available (removing variables 0 and 2)'
all_features = data_final.columns.values.tolist()
all_features.remove(variables[1])
all_features= np.array(all_features) 

# Moving the req column to the beginning of the dataframe
data_final1 = data_final.copy()
first_column = data_final1.pop(variables[1])
data_final1.insert(0, variables[1], first_column)
    
X = data_final1.values
Y = X[:, 0] 
X = X[:,[x for x in range(1,len(all_features)+1)]]

# Feature importance for the model
reg_XGBOOST = xgb.XGBRegressor()
reg_XGBOOST.fit(X, Y)
importance = pd.DataFrame(data= {'Feature': all_features, 'Score': reg_XGBOOST.feature_importances_})
importance = importance.sort_values(by= ['Score'], ascending= False)
importance.set_index('Feature', inplace= True)

# Defining the number of features to use in the models
num_features = 15       #Optimal number of features is 15  

# Features used
USE_COLUMNS = importance[:num_features].index.values
print('The features used in the XGBOOST model are:', USE_COLUMNS)

# Forecasting variable
FORECAST_COLUMN = [variables[1]]

# XGBOOST model
xtrain = data_train.loc[:, USE_COLUMNS]
xtest = data_test.loc[:, USE_COLUMNS]
ytrain = data_train.loc[:, FORECAST_COLUMN]
ytest = data_test.loc[:, FORECAST_COLUMN]
  
# Using the forecasted values of connection to forecast the req
if variables[0] in USE_COLUMNS:
    xtest[variables[0]] = df_cla.Prediction
    
reg_XGBOOST.fit(xtrain, ytrain)

# Predictions and Post-Processing
df_reg = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
df_reg['Real'] = ytest
df_reg[variables[0]] = df_cla.Prediction

df_reg['Prediction'] = np.where((df_reg['Prediction'] < 0) | (df_reg[variables[0]] == 0) & (df_reg.Prediction != 0) , 0 , df_reg['Prediction'])
df_reg[variables[0]] =  np.where((df_reg[variables[0]] == 1) & (df_reg.Prediction == 0), 0, df_reg[variables[0]])

# Plots
#Regression Plot
sns.scatterplot(data= df_reg, x= 'Real', y= 'Prediction')
plt.plot(ytest, ytest, color = "dodgerblue", linewidth= 2) 
plt.xlabel("Real energy (Wh)", alpha= 0.75, weight= "bold")
plt.ylabel("Predicted energy (Wh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.title("Correlation real vs predictions for "+ variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

#Real vs predictions in the same plot
fig,ax = plt.subplots()
ax.plot(df_reg.Real, label= "Real")
ax.plot(df_reg.Prediction, label= "Predicted", ls= '--')
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Energy (Wh)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold",fontsize= 11)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.legend(frameon= False, loc= 'best')
plt.title("Real vs predicted for "+ variables[1], alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Errors
MAE_XGBOOST = metrics.mean_absolute_error(df_reg.Real, df_reg.Prediction)
RMSE_XGBOOST = np.sqrt(metrics.mean_squared_error(df_reg.Real, df_reg.Prediction))
normRMSE_XGBOOST = 100 * RMSE_XGBOOST / ytest[variables[1]].max()
R2_XGBOOST = metrics.r2_score(df_reg.Real, df_reg.Prediction)

print('XGBOOST- Mean Absolute Error (MAE):', round(MAE_XGBOOST,2))
print('XGBOOST - Root Mean Square Error (RMSE):',  round(RMSE_XGBOOST,2))
print('XGBOOST - Normalized RMSE (%):', round(normRMSE_XGBOOST,2))
#print('XGBOOST - R square (%):', round(R2_XGBOOST,2))

###############################################################################################################################
'Forecast results'
###############################################################################################################################

pred_EV = df_reg[[variables[0]]]
pred_EV[variables[1]] = df_reg.Prediction

# Calculating the energy as acumulated
count = 0
energy_cum = 0

for i, value in enumerate(pred_EV[variables[0]]):
    # Cheking if the 'Connection' is 1, if it is, count the number of ones and sum the energy
    if value == 1:
        first_zero = False
        count += 1
        energy_cum += pred_EV[variables[1]][i]
    # Checking if the 'Connection' is 0, if it is, assign the energy_cum to the last row
    if value == 0 and first_zero is False:
        first_zero = True
        pred_EV.loc[i-count:i-1, variables[1]] = 0
        pred_EV[variables[1]][i-1] = energy_cum
        count = 0
        energy_cum = 0
    # For the end of the dataset   
    if i == len(pred_EV)-1 and value == 1:
        first_zero = True
        pred_EV.loc[i-count:i, variables[1]] = 0
        pred_EV[variables[1]][i] = energy_cum
        count = 0
        energy_cum = 0
