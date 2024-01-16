import numpy as np
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os



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




import pypyodbc 

connection_string = 'DRIVER={SQL Server};SERVER=192.168.24.177,51434;DATABASE=Ambulate;UID=sa;PWD=c0mpuc0n'

cnxn = pypyodbc.connect(connection_string)

cursor = cnxn.cursor()
cursor.execute("Select Start_Date,  From IQ_Trips")

for row in cursor:
    print('row = %r' % (row,))

cursor.close()



































