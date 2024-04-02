# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:05:12 2023
Create Databases: Consumption, EV and PV generation
@author: Herbert Amezquita
"""

'Libraries'
import pandas as pd
import mysql.connector

##############################################################################################################################
'House Power Consumption Database'
###############################################################################################################################
'Reading house power consumption data and saving it into the database'
# Power consumption data csv
df_cons = pd.read_csv('./Sample Data/Example House Cons Data.csv', parse_dates= ['Date'])
df_cons.set_index('Date', inplace=True)

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

    # Create the 'Power_cons' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Power_cons (
        Date DATETIME PRIMARY KEY,
        house_consumption FLOAT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()

    # Load the dataframe 'df_cons'
    # Iterate over the dataframe and insert data into the 'Power_cons' table
    for index, row in df_cons.iterrows():
        insert_data_query = """
        INSERT INTO Power_cons (Date, house_consumption)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            house_consumption = VALUES(house_consumption)
        """
        data = (index, row['house_consumption'])
        cursor.execute(insert_data_query, data)
        connection.commit()
    
    print("House power consumption data inserted successfully into the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed")


###############################################################################################################################
'EV House Database'
###############################################################################################################################
'Reading house EV data and saving it into the database'
# EVS connection and req data csv
df_EV = pd.read_csv('./Sample Data/Example EV Data.csv', parse_dates= ['Date'])
df_EV.set_index('Date', inplace=True)

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

    # Create the 'Power_cons' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS EV_house (
        Date DATETIME PRIMARY KEY,
        ev_connection FLOAT,
        ev_req FLOAT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()

    # Load the dataframe 'df_EV'
    # Iterate over the dataframe and insert data into the 'EV_house' table
    for index, row in df_EV.iterrows():
        insert_data_query = """
        INSERT INTO EV_house (Date, ev_connection, ev_req)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            ev_connection = VALUES(ev_connection),
            ev_req = VALUES(ev_req)
        """
        data = (index, row['ev_connection'], row['ev_req'])
        cursor.execute(insert_data_query, data)
        connection.commit()
    
    print("House EV data inserted successfully into the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed")


###############################################################################################################################
'PV Power Generation Database'
###############################################################################################################################
'Reading PV gen data and saving it into the database'
df_PV = pd.read_csv('./Sample Data/Example PV Gen Data.csv', parse_dates= ['Date'])
df_PV.set_index('Date', inplace=True)

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

    # Create the 'Power_cons' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS PV_gen (
        Date DATETIME PRIMARY KEY,
        pv_production FLOAT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()

    # Load the dataframe 'df_PV'
    # Iterate over the dataframe and insert data into the 'PV_gen' table
    for index, row in df_PV.iterrows():
        insert_data_query = """
        INSERT INTO PV_gen (Date, pv_production)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE
            pv_production = VALUES(pv_production)
        """
        data = (index, row['pv_production'])
        cursor.execute(insert_data_query, data)
        connection.commit()
    
    print("PV power generation data inserted successfully into the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed")
        
###############################################################################################################################
'Services Database'
###############################################################################################################################
# 'Reading congestion data'
# df_congestion = pd.read_csv('./Sample Data/Example Congestion.csv', parse_dates= ['Date'])
# df_congestion.set_index('Date', inplace=True)

# 'Reading wind curtailment data'
# df_windcurt = pd.read_csv('./Sample Data/Example Wind Curtailment.csv', parse_dates= ['Date'])
# df_windcurt.set_index('Date', inplace=True)
# df_windcurt.drop('Vento Médio (m/s)', axis=1, inplace= True)
# df_windcurt2 = df_windcurt.resample('15min').mean()

# 'Creating services dataframe'
# # Define the start date and frequency
# start_date = '2024-02-01 00:00'
# freq = '15T'

# # Calculate the number of rows
# if len(df_congestion) == len(df_windcurt2):
#     num_rows = len(df_congestion)
# else:
#     print('congestion and wind curtailments dataframes have different lengths!')

# timestamps = pd.date_range(start=start_date, periods=num_rows, freq=freq)

# # Create a DataFrame with the Date column
# df_services = pd.DataFrame({'Date': timestamps})
# df_services['congestion'] = df_congestion.Capacity.values
# df_services['wind_curtailment'] = df_windcurt2['Curtailed Power'].values
# df_services.set_index('Date', inplace=True)

'Reading services data and saving it into the database'
df_services = pd.read_csv('./Sample Data/Example Services Data.csv', parse_dates= ['Date'])
df_services.set_index('Date', inplace=True)
df_services.dropna(inplace= True)

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

    # Create the 'Power_cons' table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS Services (
        Date DATETIME PRIMARY KEY,
        congestion FLOAT,
        wind_curtailment FLOAT
    )
    """
    cursor.execute(create_table_query)
    connection.commit()

    # Load the dataframe 'df_services'
    # Iterate over the dataframe and insert data into the 'Services' table
    for index, row in df_services.iterrows():
        insert_data_query = """
        INSERT INTO Services (Date, congestion, wind_curtailment)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            congestion = VALUES(congestion),
            wind_curtailment = VALUES(wind_curtailment)
        """
        data = (index, row['congestion'], row['wind_curtailment'])
        cursor.execute(insert_data_query, data)
        connection.commit()
    
    print("Services data inserted successfully into the database")

except mysql.connector.Error as error:
    print(f"Error: {error}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("Database connection closed")
