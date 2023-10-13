# Taskoudis Dimitrios

import numpy as np
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os



data_dir = "C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/TestTrips/"



# Create an empty list to store the DataFrames
dataframes = []

# Get a list of all files in the folder
file_list = os.listdir(data_dir)

# Loop through the files, read CSVs, and append DataFrames to the list
for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)  # Full path to the CSV file
        df = pd.read_csv(file_path)
        dataframes.append(df)
        
        
combined_df = pd.concat(dataframes, ignore_index=True)  


# # Read the columns
# with open("C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/columns.txt", 'r') as columns_file:
#     column_names = [line.strip(' [ ] ,\n') for line in columns_file]
  
    
# # Ensure the number of column names matches the number of columns in your data (excluding the first column)
# if len(column_names) == combined_df.shape[1]:
#     # Set the column names for the DataFrame starting from the second column
#     combined_df.columns.values[0:] = column_names

# columns_to_keep = ['Trip_Num', 'Start_Date', 'Start_Lat', 'Start_Lon', 
#                    'Start_PostalCode', 'End_Date', 'End_Lat', 'End_Lon', 'End_PostalCode', 
#                    'Trip_Completed', 'age']

# dataTrips = columns_to_keep[columns_to_keep]












