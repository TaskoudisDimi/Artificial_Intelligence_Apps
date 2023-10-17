import pandas as pd
import numpy as np
import zipfile
import os


# from google.colab import drive
# drive.mount('/content/drive')

# zip_file_path = "/content/drive/MyDrive/Programming/AI/Datasets/Trips/Trips.zip"
# destination = "/content/drive/MyDrive/Programming/AI/Datasets/Trips/"

# with zipfile.ZipFile(zip_file_path, 'r') as zip:
#   zip.extractall(destination)
  


data_dir = "/content/drive/MyDrive/Programming/AI/Datasets/Trips/Trips/"


# Create an empty list to store the DataFrames
dataframes = []

# Get a list of all files in the folder
file_list = os.listdir(data_dir)


# Read the columns
with open("/content/drive/MyDrive/Programming/AI/Datasets/Trips/columns.txt", 'r') as columns_file:
    column_names = [line.strip(' [ ] ,\n') for line in columns_file]


# Loop through the files, read CSVs, and append DataFrames to the list
for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(data_dir, file)  # Full path to the CSV file
        df = pd.read_csv(file_path)
        df.columns.values[0:] = column_names
        dataframes.append(df)

# Concatenate the DataFrames vertically to append rows
combined_df = pd.concat(dataframes, axis=0, ignore_index=True)


# Correct the column names in columns_to_keep
columns_to_keep = ['Trip_Num', 'Start_Date', 'Start_Lat', 'Start_Lon', 'Start_PostalCode',
                   'End_Date', 'End_Lat', 'End_Lon', 'End_PostalCode', 'Trip_Completed', 'age']
data = combined_df[columns_to_keep]


print(data.shape)
print(list(data.columns))
print(data.sample())
print(data.info())
print(data.describe())


print(data.isnull().values.any())
rows_with_null_values = data[data.isnull().any(axis=1)]
# print(rows_with_null_values.shape)
print(rows_with_null_values.sample())

# Replaces missing values with 0 in-place
data.fillna(0,inplace=True)


# Keep only trips with Trip_Completed = 1
dataTrips = data[data['Trip_Completed'] == 1]

# Drop the column Trip_Completed
dataTrips.drop(columns=['Trip_Completed'], inplace=True)


# Check for rows with a string value in Column2 and set Column1 to 0
for index, row in dataTrips.iterrows():
    if isinstance(row['Start_PostalCode'], str):
        dataTrips.at[index, 'Start_PostalCode'] = 0

# Data Preprocessing
dataTrips['Start_PostalCode'] = dataTrips['Start_PostalCode'].astype(int)  # Convert to int

# Replace the not int values
for index, row in dataTrips.iterrows():
    try:
        int_value = int(row['Start_PostalCode'])
    except ValueError:
        # If it's not an integer, set Column1 to 0
        dataTrips.at[index, 'Start_PostalCode'] = 0


# Convert DateTime to Date format
dataTrips['Start_Date'] = pd.to_datetime(dataTrips['Start_Date'], errors='coerce')


# Drop rows with missing or invalid dates
dataTrips.dropna(subset=['Start_Date'], inplace=True)

# Feature Engineering
dataTrips['Hour'] = dataTrips['Start_Date'].dt.hour
dataTrips['Day'] = dataTrips['Start_Date'].dt.day
dataTrips['Month'] = dataTrips['Start_Date'].dt.month
dataTrips['Year'] = dataTrips['Start_Date'].dt.year


dataTrips['pickups'] = 1
dataTrips=dataTrips.groupby(['Start_Date', 'Start_PostalCode', 'Day', 'Month','Year' ])['pickups'].sum().reset_index()


dataTrips = dataTrips[dataTrips['Start_PostalCode'] != 0]


# Specify the file path where you want to save the CSV file
file_path = '/content/drive/MyDrive/Programming/AI/Datasets/Trips/output_data.csv'

# Write the data to a CSV file
dataTrips.to_csv(file_path, index=False)


# print(dataTrips.sample)

# filtered_data = dataTrips[dataTrips['pickups'] > 10]
# print(filtered_data)
print(max(dataTrips['pickups']))























