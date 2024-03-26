# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:04:15 2024
Congestion Service Forecasting
@author: Herbert Amezquita 
"""

###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

###############################################################################################################################
'Functions'
###############################################################################################################################
'Function to define the season based on the month'
def define_season(month_number):
    if month_number in [1,2,3]:
        return 1
    elif month_number in [4,5,6]:
        return 2
    elif month_number in [7,8,9]:
        return 3
    elif month_number in [10,11,12]:
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
    df['Month'] = df['Date'].dt.month   #Value: 1-12
    df['Dayofmonth'] = df['Date'].dt.day   #Value: 1-30/31
    df['Dayofweek']= df['Date'].dt.weekday+1     #Value: 1-7 (Monday-Sunday)
    df['Weekend']= np.where((df['Dayofweek']==6) | (df['Dayofweek']==7), 1, 0)    #Value: 1 if weekend, 0 if not
    df['Season']= df.Month.apply(define_season)    #Value 1-4 (winter, spring, summer and fall)    
    df['Hour'] = df['Date'].dt.hour
    df['Hour']= (df['Hour']+24).where(df['Hour']==0, df['Hour'])    #Value: 1-24
    df['Minute']= df['Date'].dt.minute     #Value: 0, 15, 30 or 45
    df.drop(['Date'], axis=1, inplace= True)
    
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
    df['Season_x']= np.cos(df['Season']*2*np.pi/4)
    df['Season_y']= np.sin(df['Season']*2*np.pi/4)
    df['Month_x']= np.cos(df['Month']*2*np.pi/12)
    df['Month_y']= np.sin(df['Month']*2*np.pi/12)
    df['Dayofmonth_x']= np.cos(df['Dayofmonth']*2*np.pi/31)
    df['Dayofmonth_y']= np.sin(df['Dayofmonth']*2*np.pi/31)
    df['Dayofweek_x']= np.cos(df['Dayofweek']*2*np.pi/7)
    df['Dayofweek_y']= np.sin(df['Dayofweek']*2*np.pi/7)
    df['Hour_x']= np.cos(df['Hour']*2*np.pi/24)
    df['Hour_y']= np.sin(df['Hour']*2*np.pi/24)
    df['Minute_x']= np.cos(df['Minute']*2*np.pi/45)
    df['Minute_y']= np.sin(df['Minute']*2*np.pi/45)
    df.drop(['Weekofyear','Dayofyear','Season','Month','Dayofmonth', 'Dayofweek','Hour', 'Minute'], axis= 1, inplace= True)
    
    return df

###############################################################################################################################
'#############################################Congestion Service Forecasting##################################################'
###############################################################################################################################
"""
Produces the congestion service forecasts for the next_hours defined

Args:
    data - dataframe containing the capacity and weather data (including training and test)
    var - name of the variable to forecast
    start_forecast - start of the forecast timestamp
    
Returns:
    pred_congestion - dataframe containing the congestion service predictions 
"""

def forecasting(data, var, start_forecast):
    print('Forecast variable: ', var)
    
    # Creating a copy of the input dataframe
    df_final = data.copy()
    
    # Creating date/time features using datetime column Date as index
    df_final = create_features(df_final)
 
    # Creating lag features of power for 2,3,4 and 5 days before
    df_final = lag_features(df_final,[2,3,4,5], var)
    df_final.dropna(inplace=True, subset= df_final.columns[1:])
    
    # Transforming date/time features into two dimensional features
    df_final = cyclical_features(df_final)
    
    # Defining training and test dataframes
    data_train = df_final.loc[: start_forecast - timedelta(minutes= 15)]
    data_test = df_final.loc[start_forecast :]
    
    # Array containing the names of all features available
    all_features = df_final.columns.values.tolist()
    all_features.remove(var)
    all_features= np.array(all_features) 
    
    # Feature importance for the model
    X = data_train.values
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
    
    # Defining the number of features to use
    num_features = 10     # Optimal number of features is 10
    
    #Features used
    USE_COLUMNS = importance[:num_features].index.values
    
    # Forecasting variable
    FORECAST_COLUMN = [var]
    
    print(f'The features used in the Random Forest model for {var} are:', USE_COLUMNS)
    
    # Random Forest model
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
        
    reg_RF.fit(xtrain, np.ravel(ytrain))
    
    # Predictions and post-processing
    pred_congestion = pd.DataFrame(reg_RF.predict(xtest), columns= [var], index= xtest.index)
    pred_congestion[var]= np.where(pred_congestion[var] > 0.3, 1, 0)

    print(f'Output: {var} predictions successfully generated!')
    
    return pred_congestion

