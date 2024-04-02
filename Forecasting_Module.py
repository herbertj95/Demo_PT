# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:35:12 2023
Demo PT Forecasting Module
@author: Herbert Amezquita 
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import pandas as pd
import mysql.connector
import sys
import warnings

warnings.filterwarnings("ignore")

###############################################################################################################################
print('######################################Demo PT - Forecasting Module########################################')
###############################################################################################################################
'Inputs'
print('Inputs')

# Define the names and the variable(s) to forecast, inside 'forecast_var' list
var1 = 'ev_connection'
var2 = 'ev_req'
var3 = 'house_consumption'
var4 = 'pv_production'
var5 = 'congestion'
var6 = 'wind_curtailment'

forecast_var = [var1, var2, var3, var4, var5, var6]
print(f'Forecasting variable(s): {forecast_var}')

###############################################################################################################################
'Getting weather forecast data from OpenWeather'
print('####################################Weather Forecast Data######################################')
import API_OpenMeteo
###############################################################################################################################
print('######################################Forecasting Info#########################################')
# Time horizon of the forecast (hours). It is defined in 'API_OpenMeteo.py'
horizon_forecast = API_OpenMeteo.next_hours
print(f'Time horizon of the forecast: {horizon_forecast} hours')

# Start and end of the forecast timestamps. Comes from 'API_OpenMeteo.py'
start_forecast = API_OpenMeteo.weather_forecast_15min.index[0]
end_forecast = API_OpenMeteo.weather_forecast_15min.index[-1]

print(f'Period to forecast: {start_forecast} to {end_forecast}')

###############################################################################################################################
'Getting historical weather data from the database'
print('###################################Historical Weather Data#####################################')
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

    # SQL query to retrieve the historical data from 'Demo_Weather' table
    query = f"SELECT * FROM Demo_Weather WHERE Date < '{start_forecast}'"

    # Creating a dataframe with the historical data 
    historical_weather = pd.read_sql(query, connection)
    historical_weather.set_index('Date', inplace= True)
    print("Historical weather data successfully loaded from Demo_Weather database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        connection.close()
        print("Database connection closed")

###############################################################################################################################
'Forecast of EV connection and req (var1 and var2)'
###############################################################################################################################
if var1 in forecast_var and var2 not in forecast_var or var1 not in forecast_var and var2 in forecast_var:
    print(f'To forecast EV connection and req you need to include both var1[{var1}] and var2[{var2}] in forecast_var')
    sys.exit()

if var1 and var2 in forecast_var:
    print('#########################Forecast of House EV Connection and Req###############################')
    import House_EV
    
    # Getting EV data from the database
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
        query = f"SELECT * FROM ist1100758.EV_house WHERE Date < '{start_forecast}'"

        # Creating a dataframe with the data 
        EV_data = pd.read_sql(query, connection)
        EV_data.set_index('Date', inplace= True)
        print("EV data successfully loaded from EV_house database")

    except mysql.connector.Error as error:
        print(f"Error: {error}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed")

    # Creating train and test dataframes
    df_train_EV = pd.merge(EV_data, historical_weather, left_index=True, right_index=True)
    df_test_EV = API_OpenMeteo.weather_forecast_15min
    
    # Creating final dataframe (training + forecast)
    df_final_EV = df_train_EV.append(df_test_EV)
    
    # Running the forecasting in 'House_EV.py'
    forecast_EV = House_EV.forecasting(df_final_EV, [var1, var2], start_forecast)

###############################################################################################################################
'Forecast of house power consumption (var3)'
###############################################################################################################################
if var3 in forecast_var:
    print('###########################Forecast of House Power Consumption#################################')
    import House_Power_cons
    
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
    
        # SQL query to retrieve the data from the 'Power_cons' table
        query = f"SELECT * FROM ist1100758.Power_cons WHERE Date < '{start_forecast}'"
    
        # Creating a dataframe with the historical house power consumption data 
        historical_cons = pd.read_sql(query, connection)
        historical_cons.set_index('Date', inplace= True)
        print("Power consumption data successfully loaded from Power_cons database")
    
    except mysql.connector.Error as error:
        print(f"Error: {error}")
    
    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed")
    
    # Creating train and test dataframes
    df_train_cons = pd.merge(historical_cons, historical_weather, left_index=True, right_index=True)
    df_test_cons = API_OpenMeteo.weather_forecast_15min
    
    # Creating final dataframe (training + forecast)
    df_final_cons = df_train_cons.append(df_test_cons)
    
    # Running the forecasting in 'House_Power_cons.py'
    forecast_house_cons = House_Power_cons.forecasting(df_final_cons, var3, start_forecast)
    
###############################################################################################################################
'Forecast of PV power generation (var4)'
###############################################################################################################################
if var4 in forecast_var:
    print('#############################Forecast of PV Power Generation###################################')
    import House_PV_gen
    
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
    
        # SQL query to retrieve the data from the 'PV_gen' table
        query = f"SELECT * FROM ist1100758.PV_gen WHERE Date < '{start_forecast}'"

        # Creating a dataframe with the data 
        historical_PV = pd.read_sql(query, connection)
        historical_PV.set_index('Date', inplace= True)
        print("PV power generation data successfully loaded from PV_gen database")

    except mysql.connector.Error as error:
        print(f"Error: {error}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed")
    
    # Creating train and test dataframes
    df_train_PV = pd.merge(historical_PV, historical_weather, left_index=True, right_index=True)
    df_test_PV = API_OpenMeteo.weather_forecast_15min
    
    # Creating final dataframe (training + forecast)
    df_final_PV = df_train_PV.append(df_test_PV)
    
    # Running the forecasting in 'House_Power_cons.py'
    forecast_PV_gen = House_PV_gen.forecasting(df_final_PV, var4, start_forecast)
    
###############################################################################################################################
'Forecast of congestion service (var5)'
###############################################################################################################################
if var5 in forecast_var:
    print('##############################Forecast of Congestion Service###################################')
    import House_Congestion
    
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
    
        # SQL query to retrieve the data from the 'Services' table
        query = f"SELECT Date, congestion FROM ist1100758.Services WHERE Date < '{start_forecast}'"

        # Creating a dataframe with the data 
        historical_congestion = pd.read_sql(query, connection)
        historical_congestion.set_index('Date', inplace= True)
        print("Congestion data successfully loaded from Services database")

    except mysql.connector.Error as error:
        print(f"Error: {error}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed")
    
    # Creating train and test dataframes
    df_train_congestion = pd.merge(historical_congestion, historical_weather, left_index=True, right_index=True)
    df_test_congestion = API_OpenMeteo.weather_forecast_15min
    
    # Creating final dataframe (training + forecast)
    df_final_congestion = df_train_congestion.append(df_test_congestion)
    
    # Running the forecasting in 'House_Congestion.py'
    forecast_congestion = House_Congestion.forecasting(df_final_congestion, var5, start_forecast)
    
###############################################################################################################################
'Forecast of wind curtailment service (var6)'
###############################################################################################################################
if var6 in forecast_var:
    print('###########################Forecast of Wind Curtailment Service################################')
    import House_Wind_curtailment  
    
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
    
        # SQL query to retrieve the data from the 'Services' table
        query = f"SELECT Date, wind_curtailment FROM ist1100758.Services WHERE Date < '{start_forecast}'"

        # Creating a dataframe with the data 
        historical_curtailment = pd.read_sql(query, connection)
        historical_curtailment.set_index('Date', inplace= True)
        print("Wind curtailment data successfully loaded from Services database")

    except mysql.connector.Error as error:
        print(f"Error: {error}")

    finally:
        if connection.is_connected():
            connection.close()
            print("Database connection closed")
    
    # Creating train and test dataframes
    df_train_curtailment = pd.merge(historical_curtailment, historical_weather, left_index=True, right_index=True)
    df_test_curtailment = API_OpenMeteo.weather_forecast_15min
    
    # Creating final dataframe (training + forecast)
    df_final_curtailment = df_train_curtailment.append(df_test_curtailment)
    
    # Running the forecasting in 'House_Wind_curtailment.py'
    forecast_curtailment = House_Wind_curtailment.forecasting(df_final_curtailment, var6, start_forecast)
    
###############################################################################################################################
'Forecasting Results'
print('####################################Forecasting Results########################################')
###############################################################################################################################
# Creating dataframe for the predictions
df_forecast =  pd.DataFrame(index= pd.date_range(start= start_forecast, end= end_forecast, freq='15min')) 
df_forecast.index.name= 'Date'

# Adding EV connection and req
if var1 and var2 in forecast_var:
    df_forecast = df_forecast.join(forecast_EV)
    
# Adding house power consumption 
if var3 in forecast_var:
    df_forecast = df_forecast.join(forecast_house_cons)
    
# Adding PV power generation
if var4 in forecast_var:
    df_forecast = df_forecast.join(forecast_PV_gen)
    
# Adding congestion service
if var5 in forecast_var:
    df_forecast = df_forecast.join(forecast_congestion)
    
# Adding wind curtailment service
if var6 in forecast_var:
    df_forecast = df_forecast.join(forecast_curtailment)
    
print('The forecasts can be found in df_forecast')

# Saving the predictions into the database
if var1 and var2 and var3 and var4 and var5 and var6 in forecast_var:
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
    
        # Create a cursor object to interact with the database
        cursor = connection.cursor()
    
        # Create the 'Demo_Forecast' table if it doesn't exist
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS Demo_Forecast (
            Date DATETIME PRIMARY KEY,
            {var1} FLOAT,
            {var2} FLOAT,
            {var3} FLOAT,
            {var4} FLOAT,
            {var5} FLOAT,
            {var6} FLOAT
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
        
        # Delete data that is already in table 'Demo_Forecast'
        # truncate_table_query = """
        # TRUNCATE TABLE Demo_Forecast
        # """
        # cursor.execute(truncate_table_query)
        # connection.commit()
        
        # Load the dataframe 'df_forecast' updating values that already exist
        # Iterate over the dataframe and insert data into the 'Forecast' table
        for index, row in df_forecast.iterrows():
            insert_data_query = f"""
            INSERT INTO Demo_Forecast (Date, {var1}, {var2}, {var3}, {var4}, {var5}, {var6})
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                {var1} = VALUES({var1}),
                {var2} = VALUES({var2}),
                {var3} = VALUES({var3}),
                {var4} = VALUES({var4}),
                {var5} = VALUES({var5}),
                {var6} = VALUES({var6})
            """
            data = (index, float(row[var1]), float(row[var2]), float(row[var3]), float(row[var4]), float(row[var5]), float(row[var6]))
            cursor.execute(insert_data_query, data)
            connection.commit()
        
        print(f'Forecasts of {forecast_var} for next {horizon_forecast} hours inserted successfully into Demo_Forecast database')
    
    except mysql.connector.Error as error:
        print(f"Error: {error}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Database connection closed")

else:
    print('To save the predictions into the forecast database you need to incluye all variables (var1-var6)')
        