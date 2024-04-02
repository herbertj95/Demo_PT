# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:19:22 2024
House Congestion Service Forecast
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import DateFormatter
import mysql.connector

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


'############################################House Congestion Service Forecast################################################'

###############################################################################################################################
'Inputs'
###############################################################################################################################
start_forecast = datetime.strptime('2024-04-01', '%Y-%m-%d')
end_forecast = datetime.strptime('2024-04-02', '%Y-%m-%d')
var = 'congestion'

###############################################################################################################################
'Getting power data from the database'
print('###################################Congestion data######################################')
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
    query = "SELECT Date, congestion FROM ist1100758.Services"

    # Creating a dataframe with the data 
    congestion = pd.read_sql(query, connection)
    congestion.set_index('Date', inplace= True)
    print("Congestion data successfully loaded from the database")

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
    query = "SELECT * FROM Demo_Weather"

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
# Merging congestion data with weather data
df_final = pd.merge(congestion, historical_weather, left_index=True, right_index=True)

# Creating date/time features using datetime column Date as index
df_final = create_features(df_final)

# Creating lag features of power for 2,3,4 and 5 days before
df_final = lag_features(df_final,[2,3,4,5], var)
df_final.dropna(inplace=True, subset= df_final.columns[1:])

# Transforming date/time features into two dimensional features
df_final = cyclical_features(df_final)

###############################################################################################################################
'Feature selection'
###############################################################################################################################
# Correlation matrix
corr = df_final.corr()[[var]].sort_values(by= var, ascending= False).round(2)

# Heatmap feature correlation
fig = plt.subplots(figsize=(12, 10))
heatmap = sns.heatmap(corr[1:], vmin= -1, vmax= 1, annot= True, cmap= 'BrBG')
heatmap.set_title('Features correlation with congestion power', alpha= 0.75, weight= "bold", pad= 10)
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 16)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 16)
plt.show()

# Array containing the names of all features available
all_features = df_final.columns.values.tolist()
all_features.remove(var)
all_features= np.array(all_features) 

# Feature importance for the model
X = df_final.values
Y = X[:,0] 
X = X[:,[x for x in range(1,len(all_features)+1)]]

parameters_RF= {'bootstrap': True,
                  'min_samples_leaf': 3,
                  'n_estimators': 200, 
                  'min_samples_split': 7,
                  'max_depth': 30,
                  'max_leaf_nodes': None,
                  'random_state': 18}
      
reg_RF= RandomForestRegressor(**parameters_RF)
reg_RF.fit(X, Y)
importance= pd.DataFrame(data= {'Feature': all_features, 'Score': reg_RF.feature_importances_})
importance= importance.sort_values(by=['Score'], ascending= False)
importance.set_index('Feature', inplace=True)

###############################################################################################################################
'Forecasting'
###############################################################################################################################

# Defining the number of features to use
num_features = 10    # Optimal number of features is 10

# Defining training and test dataframes
data_train = df_final.loc[: start_forecast - timedelta(minutes= 15)]
data_test = df_final.loc[start_forecast : end_forecast]

# Plot train-test
fig,ax = plt.subplots()
coloring = df_final[var].max()
plt.plot(data_train.index, data_train[var], color= "darkcyan", alpha= 0.75)
plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
plt.plot(data_test.index, data_test[var], color = "dodgerblue", alpha= 0.60)
plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Power (MW)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75,weight= "bold", rotation= 45)
plt.yticks(alpha= 0.75,weight= "bold")
plt.title(" Train - Test split for congestion", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

#Features used
USE_COLUMNS = importance[:num_features].index.values

# Forecasting variable
FORECAST_COLUMN = [var]

print(f'The features used in the Random Forest model for {var} are:', USE_COLUMNS)

# Random Forest model
xtrain = data_train.loc[:, USE_COLUMNS]
xtest = data_test.loc[:, USE_COLUMNS]
ytrain = data_train.loc[:, FORECAST_COLUMN]
ytest = data_test.loc[:, FORECAST_COLUMN]
    
reg_RF.fit(xtrain, np.ravel(ytrain))

# Predictions and post-processing
pred_congestion = pd.DataFrame(reg_RF.predict(xtest), columns= ['Prediction'], index= xtest.index)
pred_congestion['Real'] = ytest

# Regression plot
sns.scatterplot(data= pred_congestion, x='Real', y= 'Prediction')
plt.plot(ytest, ytest, color = "red", linewidth= 1) 
plt.xlabel("Real Power (MW)", alpha= 0.75, weight= "bold")
plt.ylabel("Predicted Power (MW)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold")
plt.yticks(alpha= 0.75, weight= "bold")
plt.title("Correlation real vs predictions for congestion", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Real vs predictions in the same plot
fig, ax = plt.subplots()
sns.lineplot(x= pred_congestion.index, y= pred_congestion.Real, label= "Real", ax= ax, linestyle = "dashed")
sns.lineplot(x= pred_congestion.index, y= pred_congestion.Prediction, label= "Predicted", ax= ax, linewidth = 0.5)
plt.gca().xaxis.set_major_formatter(date_format)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Power (MW)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", rotation= 45)
plt.yticks(alpha= 0.75, weight= "bold")
plt.legend(loc= 'upper left')
plt.title("Real vs predicted congestion", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
plt.show()

# Errors
MAE = metrics.mean_absolute_error(pred_congestion.Real, pred_congestion.Prediction)
RMSE = np.sqrt(metrics.mean_squared_error(pred_congestion.Real, pred_congestion.Prediction))
normRMSE = 100 * RMSE/ ytest[var].max()
R2 = metrics.r2_score(pred_congestion.Real, pred_congestion.Prediction)

print('Mean Absolute Error (MAE):', round(MAE,2))
print('Root Mean Square Error (RMSE):',  round(RMSE,2))
print('Normalized RMSE (%):', round(normRMSE,2))
print('R square (%):', round(R2,2))

'Converting predictions into 0 and 1'
pred_congestion[var]= np.where(pred_congestion['Prediction'] > 0.3, 1, 0)
