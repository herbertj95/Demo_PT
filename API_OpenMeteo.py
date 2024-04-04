# -*- coding: utf-8 -*-
"""
Created on Mar 28 2024
Open-Meteo API
@author: Herbert Amezquita
"""

###############################################################################################################################
'Libraries'
###############################################################################################################################
import openmeteo_requests
import requests_cache
import numpy as np
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import pytz
import mysql.connector

###############################################################################################################################
'Auxiliary funtion round_to_next_15_minutes'
###############################################################################################################################
"""
Rounds the given timestamp to the next 15-minute interval.

Parameters:
    current_time (datetime): The timestamp to be rounded.

Returns:
    datetime: The rounded timestamp to the next 15-minute interval.
"""
def round_to_next_15_minutes(current_time):
    # Calculate the next 15-minute interval
    rounded_minute = (current_time.minute // 15 + 1) * 15
    if rounded_minute == 60:
        rounded_time = current_time + timedelta(hours=1) - timedelta(minutes=current_time.minute, seconds=current_time.second)
    else:
        rounded_time = current_time + timedelta(minutes=rounded_minute - current_time.minute, seconds=-current_time.second)
        
    # Truncate the seconds and microseconds
    rounded_time = rounded_time.replace(second=0, microsecond=0)
    
    # Remove timezone
    rounded_time = rounded_time.replace(tzinfo=None)
    
    return rounded_time


###############################################################################################################################
'Principal funtion get_weather_data'
###############################################################################################################################
"""
Gets the historical and forecast weather data from OpenMeteo.

Parameters:
    lat: Latitude of the location.
    lon: Longitude of the location.
    next_hours: Time horizon of the forecast (in hours)
    past_days: Historical data to retrieve (in days). 

Returns:
    weather_forecast_15min: Weather forecast dataframe for the next_hours defined.
"""
def get_weather_data(lat, lon, next_hours, past_days):
    ###########################################################################################################################
    'Inputs'
    ###########################################################################################################################
    'Define lat and lon of the location'

    lat = lat
    lon = lon
    
    'Define the time horizon of the forecast (in hours)'
    next_hours = next_hours
    
    'Define the number of days in the past to retrieve (HISTORICAL WEATHER)'
    past_days = past_days
    
    'Define the number of days in the future to retrieve, including today (WEATHER FORECASTS)'
    future_days = 3
    
    'Specify your timezone'
    local_timezone = 'Europe/Lisbon'
    
    ###########################################################################################################################
    'WEATHER FORECASTS - To get the weather forecasts for the next_hours defined'
    ###########################################################################################################################
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
    	"latitude": lat,
    	"longitude": lon,
    	"minutely_15": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
    	"forecast_days": future_days,
        "models": "best_match"}
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process minutely_15 forecast data. The order of variables needs to be the same as requested
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = np.round(minutely_15.Variables(0).ValuesAsNumpy(), decimals=2)
    minutely_15_relative_humidity_2m = np.round(minutely_15.Variables(1).ValuesAsNumpy(), decimals=2)
    minutely_15_wind_speed_10m = np.round(minutely_15.Variables(2).ValuesAsNumpy(), decimals=2)
    minutely_15_wind_direction_10m = np.round(minutely_15.Variables(3).ValuesAsNumpy(), decimals=2)
    minutely_15_shortwave_radiation = np.round(minutely_15.Variables(4).ValuesAsNumpy(), decimals=2)
    minutely_15_diffuse_radiation = np.round(minutely_15.Variables(4).ValuesAsNumpy(), decimals=2)
    minutely_15_direct_normal_irradiance = np.round(minutely_15.Variables(5).ValuesAsNumpy(), decimals=2)
    
    # Creating dataframe df for the no_days defined and including the weather forecasts 
    date_range = pd.date_range(
    	start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = minutely_15.Interval()))
    
    df = pd.DataFrame(index= date_range[:-1].tz_localize(None))
    df.index.name = 'Date'
    
    df["Temperature"] = minutely_15_temperature_2m
    df["Humidity"] = minutely_15_relative_humidity_2m
    df["Wind_speed"] = minutely_15_wind_speed_10m
    df["Wind_direction"] = minutely_15_wind_direction_10m
    df["Ghi"] = minutely_15_shortwave_radiation
    df["Dhi"] = minutely_15_diffuse_radiation
    df["Dni"] = minutely_15_direct_normal_irradiance
    
    # Determine the start and the end of the forecast timestamps
    utc_now = datetime.utcnow()     # Current time in UTC
    local_now = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(local_timezone))     # Convert UTC time to local timezone
    start_forecast = round_to_next_15_minutes(local_now)
    end_forecast = start_forecast + timedelta(hours= next_hours) - timedelta(minutes= 15)
    
    # Creating the final dataframe df_forecast_15min with the weather forecasts for the next_hours defined
    weather_forecast_15min = pd.DataFrame(index= pd.date_range(start=start_forecast, end=end_forecast, freq='15min'))
    weather_forecast_15min =  pd.merge(weather_forecast_15min, df, how='left', left_index=True, right_index=True)
    
    print(f'Weather forecasts for next {next_hours} hours successfully downloaded from Open-Meteo')
    
    ###########################################################################################################################
    'HISTORICAL WEATHER - To get the past data for the past_days defined'
    ###########################################################################################################################
    # Requesting the historical data
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
    	"latitude": lat,
    	"longitude": lon,
    	"minutely_15": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"],
    	"past_days": past_days,
        "models": "best_match"}
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process minutely_15 historical data. The order of variables needs to be the same as requested
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = np.round(minutely_15.Variables(0).ValuesAsNumpy(), decimals=2)
    minutely_15_relative_humidity_2m = np.round(minutely_15.Variables(1).ValuesAsNumpy(), decimals=2)
    minutely_15_wind_speed_10m = np.round(minutely_15.Variables(2).ValuesAsNumpy(), decimals=2)
    minutely_15_wind_direction_10m = np.round(minutely_15.Variables(3).ValuesAsNumpy(), decimals=2)
    minutely_15_shortwave_radiation = np.round(minutely_15.Variables(4).ValuesAsNumpy(), decimals=2)
    minutely_15_diffuse_radiation = np.round(minutely_15.Variables(4).ValuesAsNumpy(), decimals=2)
    minutely_15_direct_normal_irradiance = np.round(minutely_15.Variables(5).ValuesAsNumpy(), decimals=2)
    
    # Creating dataframe df_train_15min with the weather data for the past_days defined
    date_range = pd.date_range(
    	start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = minutely_15.Interval()))
    
    weather_historical_15min = pd.DataFrame(index= date_range[:-1].tz_localize(None))
    weather_historical_15min.index.name = 'Date'
    
    weather_historical_15min["Temperature"] = minutely_15_temperature_2m
    weather_historical_15min["Humidity"] = minutely_15_relative_humidity_2m
    weather_historical_15min["Wind_speed"] = minutely_15_wind_speed_10m
    weather_historical_15min["Wind_direction"] = minutely_15_wind_direction_10m
    weather_historical_15min["Ghi"] = minutely_15_shortwave_radiation
    weather_historical_15min["Dhi"] = minutely_15_diffuse_radiation
    weather_historical_15min["Dni"] = minutely_15_direct_normal_irradiance
    
    # Historical data goes from start_historical until the 15min before the start of the forecast
    start_historical = start_forecast - timedelta(days= past_days)
    weather_historical_15min = weather_historical_15min[(start_historical <= weather_historical_15min.index) & (weather_historical_15min.index < start_forecast)]
    
    print(f'Historical weather data for the previous {past_days} days successfully downloaded from Open-Meteo')
    
    ###########################################################################################################################
    'Connecting to the database to save the historical weather data'
    ###########################################################################################################################
    # Define database connection parameters
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
    
        # Create the 'Demo_Weather' table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS Demo_Weather (
            Date DATETIME PRIMARY KEY,
            Temperature FLOAT,
            Humidity FLOAT,
            Wind_speed FLOAT,
            Wind_direction FLOAT,
            Ghi FLOAT,
            Dhi FLOAT,
            Dni FLOAT
        )
        """
        cursor.execute(create_table_query)
        connection.commit()
    
        # Load the dataframe 'weather_historical_15min'
        # Iterate over the dataframe and insert data into the Demo_'Weather' table
        for index, row in weather_historical_15min.iterrows():
            insert_data_query = """
            INSERT INTO Demo_Weather (Date, Temperature, Humidity, Wind_speed, Wind_direction, Ghi, Dhi, Dni)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                Temperature = VALUES(Temperature),
                Humidity = VALUES(Humidity),
                Wind_speed = VALUES(Wind_speed),
                Wind_direction = VALUES(Wind_direction),
                Ghi = VALUES (Ghi),
                Dhi = VALUES (Dhi),
                Dni = VALUES (Dni)
            """
            data = data = (index, float(row['Temperature']), float(row['Humidity']), float(row['Wind_speed']), float(row['Wind_direction']), float(row['Ghi']), float(row['Dhi']), float(row['Dni']))
            cursor.execute(insert_data_query, data)
            connection.commit()
        
        print(f"Weather data for {past_days} days before the start of the forecast inserted successfully into database Demo_Weather")
    
    except mysql.connector.Error as error:
        print(f"Error: {error}")
    
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("Database connection closed")
            
    ###########################################################################################################################
    'The return of the OpenMeteo_API funtion is the weather forecasts dataframe for the next_hours defined'
    ###########################################################################################################################
   
    return weather_forecast_15min
