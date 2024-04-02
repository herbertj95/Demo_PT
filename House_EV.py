# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:32:35 2023
House EV Connection and Req Forecasting
@author: Herbert Amezquita 
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import xgboost as xgb
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
'#########################House EV Connection (Classification) and Req (Regression) Forecasting###############################'
###############################################################################################################################
"""
Generates the house EV connection and req forecasts for the next_hours defined

Args:
    data_final - dataframe containing the EV data and weather data (including training and test)
    variables - list with the names of the variables to forecast (var1 and var2 in the main)
    start_forecast - start of the forecast timestamp
    
Returns:
    pred_EV - dataframe containing the EV connection and req predictions 
"""

def forecasting(data, variables, start_forecast):
    
    # Creating a copy of the input dataframe
    data_final = data.copy()
    
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
    # error = data_final[(data_final[variables[0]] == 0) & (data_final[variables[1]] != 0) | (data_final[variables[0]] == 1) & (data_final[variables[1]] == 0)]

    # Creating date/time features using datetime column Date as index
    data_final = create_features(data_final) 
    
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
   
     # Array containing the names of all features available (removing variables)
    all_features = data_final.columns.values.tolist()
    all_features.remove(variables[0])
    all_features.remove(variables[1])
    all_features= np.array(all_features) 
        
    X = data_train.values
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
    print(f'The features used in the XGBOOST model for {variables[0]} are:', USE_COLUMNS)
    
    # Forecasting variable
    FORECAST_COLUMN = [variables[0]]
    
    # XGBOOST model
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    
    cla_XGBOOST.fit(xtrain, ytrain)
    
    # Predictions
    df_cla = pd.DataFrame(cla_XGBOOST.predict(xtest), columns=[variables[0]], index= xtest.index)
    
    print(f'Output: {variables[0]} predictions successfully generated!')
    
    ###############################################################################################################################
    'Forecasting Energy Req (Regression)'
    print('Forecast variable: ', variables[1])
    ###############################################################################################################################

    #Array containing the names of all features available (removing variables 0 and 2)'
    all_features = data_final.columns.values.tolist()
    all_features.remove(variables[1])
    all_features= np.array(all_features) 
    
    # Moving the req column to the beginning of the dataframe
    data_train2 = data_train.copy()
    first_column = data_train2.pop(variables[1])
    data_train2.insert(0, variables[1], first_column)
        
    X = data_train2.values
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
    print(f'The features used in the XGBOOST model for {variables[1]} are:', USE_COLUMNS)
    
    # Forecasting variable
    FORECAST_COLUMN = [variables[1]]
    
    # XGBOOST model
    xtrain = data_train.loc[:, USE_COLUMNS]
    xtest = data_test.loc[:, USE_COLUMNS]
    ytrain = data_train.loc[:, FORECAST_COLUMN]
    
    # Using the forecasted values of connection to forecast the req
    if variables[0] in USE_COLUMNS:
        xtest[variables[0]] = df_cla[variables[0]]
        
    reg_XGBOOST.fit(xtrain, ytrain)
    
    # Predictions and Post-Processing
    df_reg = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= [variables[1]], index= xtest.index)
    df_reg[variables[0]] = df_cla[variables[0]]
    
    df_reg[variables[1]] = np.where((df_reg[variables[1]] < 0) | (df_reg[variables[0]] == 0) & (df_reg[variables[1]] != 0), 0, df_reg[variables[1]])
    df_reg[variables[0]] =  np.where((df_reg[variables[0]] == 1) & (df_reg[variables[1]] == 0), 0, df_reg[variables[0]])
    
    print(f'Output: {variables[1]} predictions successfully generated!')
    
    ###############################################################################################################################
    'Forecast results'
    ###############################################################################################################################
    
    pred_EV = df_reg[[variables[0]]]
    pred_EV[variables[1]] = df_reg[variables[1]]
    
    # Calculating the energy as acumulated
    count = 0
    energy_cum = 0
    
    for i, value in enumerate(pred_EV[variables[0]]):
        # Cheking if the 'Connection' is 1, if it is, count the number of ones and sum the energy
        if value == 1:
            first_zero = False
            count += 1
            energy_cum += pred_EV[variables[1]][i]
            pred_EV[variables[1]][i] = 0
        # Checking if the 'Connection' is 0, if it is, assign the energy_cum to the last row
        if value == 0 and first_zero is False:
            first_zero = True
            pred_EV[variables[1]][i-1] = energy_cum
            count = 0
            energy_cum = 0
        # For the end of the dataset   
        if i == len(pred_EV)-1 and value == 1:
            first_zero = True
            pred_EV[variables[1]][i] = energy_cum
            count = 0
            energy_cum = 0
            
    return pred_EV
