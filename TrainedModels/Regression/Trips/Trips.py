# Taskoudis Dimitrios

import numpy as np
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os

##Unzip file
# with ZipFile("E:/Trips/Trips.zip", 'r') as z:
#     z.extractall(path="E:/Trips/")

data_dir = "C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/Trips/"

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
        
        
        # # Read the CSV file in smaller chunks (adjust chunk size as needed)
        # chunk_size = 10000  # You can adjust this based on your available memory
        # chunks = pd.read_csv(file_path, chunksize=chunk_size)
        
        # for chunk in chunks:
        #     combined_df = combined_df.append(chunk, ignore_index=True)
        
        
combined_df = pd.concat(dataframes, ignore_index=True)  
        

# #Read the first data
# data2016 = pd.read_csv("C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/Trips/2016Trips.csv")


# # Read the columns
# with open("C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/columns.txt", 'r') as columns_file:
#     column_names = [line.strip(' [ ] ,\n') for line in columns_file]
  
    
# # Ensure the number of column names matches the number of columns in your data (excluding the first column)
# if len(column_names) == data2016.shape[1]:
#     # Set the column names for the DataFrame starting from the second column
#     data2016.columns.values[0:] = column_names


# # Read the cleaned columns
# with open("C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/cleanColumns.txt", 'r') as columns_file:
#     column_names_cleaned = [line.strip(' [ ] ,\n') for line in columns_file]


# dataTrips = data2016.filter(items=column_names_cleaned)


# # print(dataTrips.shape)
# # print(list(dataTrips.columns))
# # print(dataTrips.sample())
# # print(dataTrips.info())
# # print(dataTrips.describe())


# print(dataTrips.isnull().values.any())
# rows_with_null_values = dataTrips[dataTrips.isnull().any(axis=1)]
# print(rows_with_null_values.shape)
# print(rows_with_null_values.sample())

# # Replaces missing values with 0 in-place
# dataTrips.fillna(0,inplace=True)


# # Keep only trips with Trip_Completed = 1
# dataTrips = dataTrips[dataTrips['Trip_Completed'] == 1]

# dataTrips.drop(columns=['Trip_Completed'], inplace=True)


# # countTrips_Date = dataTrips.groupby('Start_Date').sum().reset_index()
# # countTrips_Zip = dataTrips.groupby('Start_PostalCode').sum().reset_index()


# # Check for rows with a string value in Column2 and set Column1 to 0
# for index, row in dataTrips.iterrows():
#     if isinstance(row['Start_PostalCode'], str):
#         dataTrips.at[index, 'Start_PostalCode'] = 0

# # Data Preprocessing
# dataTrips['Start_PostalCode'] = dataTrips['Start_PostalCode'].astype(int)  # Convert to int

# # Replace the not int values
# for index, row in dataTrips.iterrows():
#     try:
#         int_value = int(row['Start_PostalCode'])
#     except ValueError:
#         # If it's not an integer, set Column1 to 0
#         dataTrips.at[index, 'Start_PostalCode'] = 0
        


# # Convert DateTime to Date format
# dataTrips['Start_Date'] = pd.to_datetime(dataTrips['Start_Date'], errors='coerce')


# # Drop rows with missing or invalid dates
# dataTrips.dropna(subset=['Start_Date'], inplace=True)

# # Feature Engineering
# dataTrips['Hour'] = dataTrips['Start_Date'].dt.hour
# dataTrips['Day'] = dataTrips['Start_Date'].dt.day
# dataTrips['Month'] = dataTrips['Start_Date'].dt.month
# dataTrips['Year'] = dataTrips['Start_Date'].dt.year


# dataTrips['pickups'] = 1
# dataTrips=dataTrips.groupby(['Start_Date', 'Start_PostalCode', 'Day', 'Month','Year' ])['pickups'].sum().reset_index()


# dataTrips = dataTrips[dataTrips['Start_PostalCode'] != 0]


# Plot the data
# plt.figure(figsize=(15,8))
# plt.bar(dataTrips['Start_Date'], dataTrips['pickups'])

# plt.xlabel("Date")
# plt.ylabel("Count")

# plt.tight_layout()
# plt.show()






# ## Build AI Model
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error


# X = dataTrips[['Start_PostalCode', 'Day', 'Month', 'Year']]
# y = dataTrips['pickups']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

# # model = LinearRegression()
# # model.fit(X_train, y_train)

# # y_pred = model.predict(X_test)

# # # Assuming y_test contains the actual values and y_pred contains the predicted values
# # mae = mean_absolute_error(y_test, y_pred)
# # mse = mean_squared_error(y_test, y_pred)
# # rmse = mean_squared_error(y_test, y_pred, squared=False)  # Pass squared=False to get RMSE
# # r2 = r2_score(y_test, y_pred)

# # print(f"Mean Absolute Error (MAE): {mae}")
# # print(f"Mean Squared Error (MSE): {mse}")
# # print(f"Root Mean Squared Error (RMSE): {rmse}")
# # print(f"R-squared (R2) Score: {r2}")


# # Create and Train the Random Forest Regressor Model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Make Predictions
# y_pred = rf_model.predict(X_test)

# # Evaluate the Model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R-squared (R2) Score: {r2}")


# # Save the model
# import joblib 

# joblib.dump(rf_model, "C:/Users/chris/Desktop/Dimitris/Tutorials/Python/Trips/Linear_regression_model.pkl")








# Display map
# import folium

# # Create a base map
# m = folium.Map(location=[38.1458392, 24.4813,], zoom_start=10)

# # You can add markers, polygons, and other elements to the map as needed

# # Save the map as an HTML file or display it in a Jupyter Notebook
# m.save("my_map.html")  # Save the map to an HTML file
# m  # Display the map in a Jupyter Notebook


# # Open the map in your default web browser
# import webbrowser
# webbrowser.open("my_map.html", new=2)




























