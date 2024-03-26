# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:31:53 2023
OpenWeather API
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import requests
import certifi
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date
import mysql.connector

###############################################################################################################################
'Inputs'
###############################################################################################################################
'Define lat and lon of the location'
# This one is Ponta Delgada in Sao Miguel, Acores
lat = '37.7412'
lon = '-25.6756'

'OpenWeather api key'
API_key = '1715575eec2afb64b323673890032594'

'Define the time horizon of the weather forecasts to retrieve (in hours)'
next_hours = 36

#print('#######################################OpenWeather API#########################################')

###############################################################################################################################
'WEATHER FORECASTS API - Used to predict the next_hours defined'
###############################################################################################################################
'OpenWeather'
# URL to get the hourly weather forecast for the next 4 days
url_forecast = f'https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&appid={API_key}'

# Doing the call
response = requests.get(url_forecast, verify= certifi.where())

# Checking the response
# print(response.text)

# Data output as JSON
data = response.json()

# Dictionary with the weather forecast for each date {key=date, values= [temp, humidity, wind speed, wind dir, clouds]}
values = dict()
for index, key in enumerate(data['list']):
    # Taking the weather forecast for the x next_hours defined 
    if index == next_hours+1:
        break
    dt = datetime.fromtimestamp(key['dt'])
    values[dt]= [key['main']['temp'], key['main']['humidity'], key['wind']['speed'], key['wind']['deg'], key['clouds']['all']]
    
# Dataframe with the final hourly weather forecasts for the next_hours
df_forecast = pd.DataFrame.from_dict(values, orient='index', columns=['Temperature', 'Humidity', 'Wind_speed', 'Wind_direction', 'Cloudiness'])
df_forecast.index.name = 'Date'

'Radiation (Azores webpage)'
# URL of the webpage
url_rad = "https://pt.tutiempo.net/radiacao-solar/ponta-delgada-nordela-acores.html"

# Fetch the webpage content
response = requests.get(url_rad)
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find divs with class "RadiacionSolar"
    radiation_divs = horas_dia_divs = soup.find_all("div", id=lambda x: x and x.startswith("HorasDia"))
    
    # Dictionary of timestamps and radiation values
    radiation = dict()
    
    # Extract solar radiation values for today and next 3 days
    for index, radiation_div in enumerate(radiation_divs[:4], start=1):     
        # Create the date
        if index == 1:      # Today
            date = date.today()
            date_start = date
        elif index == 2:    # Tomorrow
            date = date_start + timedelta(days= 1)
        elif index == 3:    # Day after tomorrow
            date = date_start + timedelta(days= 2)
        elif index == 4:    # Two days after tomorrow
            date = date_start + timedelta(days= 3)
            date_end = date + timedelta(days= 1)
            
        # Extract individual hourly solar radiation values
        hourly_radiation_values = radiation_div.find_all('div', class_='horhor')
        
        # print(f"Date: {date}")
        for hourly_radiation in hourly_radiation_values:
            hour = int(hourly_radiation.find('span', class_='hora').text.split(':')[0].strip())
            radiation_value = int(hourly_radiation.find('strong').text.strip())
            # print(f"Hour: {hour}, Radiation Value: {radiation_value} W/m2")
            timestamp = datetime(date.year, date.month, date.day, hour)
            radiation[timestamp] = radiation_value
            
    # Create radiation dataframe
    date_range = pd.date_range(start= date_start, end= date_end, freq='1H')
    df_radiation =  pd.DataFrame(index= date_range[:-1], columns=['Ghi'])
    df_radiation.index.name = 'Date'
    df_radiation['Ghi'] = 0   
    
    for timestamp, value in radiation.items():
        if timestamp in df_radiation.index:
            df_radiation.at[timestamp, 'Ghi'] = value

else:
    print("Failed to fetch webpage.")
    
'Final dataframe'
df_forecast['Ghi'] = df_radiation['Ghi']

# Resampling to 15min data resolution using ffill
df_forecast_15min = df_forecast.resample('15min').ffill()

# Removing the last row
df_forecast_15min.drop(df_forecast_15min.index[-1], inplace= True)

# Saving the weather forecasts data in a csv
# df_forecast_15min.to_csv('Weather Forecasts.csv', index= True) 
# print('csv with the weather forecasts saved')

print(f'Weather forecasts for next {next_hours} hours successfully downloaded in API_OpenWeather.df_forecast_15min from OpenWeather')

###############################################################################################################################
'HISTORICAL API - Used to train the forecast model'
###############################################################################################################################
# Specify the ending timestamp of the training (Option 2)
#end_date = datetime(2023, 9, 12, 0, 0, 0)

# The end date corresponds to the timestamp of the hour before the start of the forecast (Option 1)
end_date = df_forecast.index[0] - timedelta(hours=1)

# The start date corresponds to the timestamp one week before the end date
start_date = end_date - timedelta(days=7)

# Transforming start and end dates to Unix Timestamp
start = int(start_date.timestamp())
end = int(end_date.timestamp())

# URL to get the hourly weather data for the week before the start of the forecast
url_historic= f'https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={API_key}'

# Doing the call
response = requests.get(url_historic, verify= certifi.where())

# Checking the response
# print(response.text)

# Data output as JSON
data = response.json()

# Dictionary with the weather data for each date {key=date, values= [temp, humidity, wind speed, wind dir, clouds]}
values = dict()
for index, key in enumerate(data['list']):
    date = datetime.fromtimestamp(key['dt'])
    values[date]= [key['main']['temp'], key['main']['humidity'], key['wind']['speed'], key['wind']['deg'], key['clouds']['all']]

# Dataframe with the final hourly weather data for the week before the start of the forecast
df_train = pd.DataFrame.from_dict(values, orient='index', columns=['Temperature', 'Humidity', 'Wind_speed', 'Wind_direction', 'Cloudiness'])
df_train.index.name = 'Date'

# Resampling to 15min data resolution using ffill
df_train_15min = df_train.resample('15min').ffill()

# Creating the additional timestamps of the last hour
date_range = pd.date_range(start= df_train_15min.index[-1] + timedelta(minutes=15), end= df_train_15min.index[-1] + timedelta(minutes=45), freq='15min')
additional_rows = pd.DataFrame(index= date_range, columns= df_train_15min.columns)

# Concatenating the additional rows with the final dataframe and filling them using ffill
df_train_15min = pd.concat([df_train_15min, additional_rows])
df_train_15min.fillna(method='ffill', inplace=True)

# Saving the historical weather data in a csv
# df_train_15min.to_csv('Weather Historical.csv', index= True)
# print('csv with the historical weather data saved')

#print('Weather data for one week before the start of the forecast successfully downloaded in df_train_15min from OpenWeather')
 
###############################################################################################################################
'Connecting to the database to save the historical weather data'
###############################################################################################################################
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

    # Create the 'Weather' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Weather (
        Date DATETIME PRIMARY KEY,
        Temperature FLOAT,
        Humidity FLOAT,
        Wind_speed FLOAT,
        Wind_direction FLOAT,
        Cloudiness FLOAT,
        Ghi FLOAT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()

    # Load the dataframe 'df_train_15min'
    # Iterate over the dataframe and insert data into the 'Weather' table
    for index, row in df_train_15min.iterrows():
        insert_data_query = """
        INSERT INTO Weather (Date, Temperature, Humidity, Wind_speed, Wind_direction, Cloudiness, Ghi)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            Temperature = VALUES(Temperature),
            Humidity = VALUES(Humidity),
            Wind_speed = VALUES(Wind_speed),
            Wind_direction = VALUES(Wind_direction),
            Cloudiness = VALUES(Cloudiness),
            Ghi = VALUES (Ghi)
        """
        data = (index, row['Temperature'], row['Humidity'], row['Wind_speed'], row['Wind_direction'], row['Cloudiness'], 0)
        cursor.execute(insert_data_query, data)
        connection.commit()
    
    print("Weather data for one week before the start of the forecast inserted successfully into the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed")
