import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedianHouseValue')


X_Train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

model = LinearRegression()
model.fit(X_Train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Example: Predict the median house value for a specific input
new_input = np.array([[8.3252, 41.0, 6.98412698, 1.02380952, 322.0, 2.55555556, 37.88, -122.23]])
predicted_value = model.predict(new_input)
print(f"Predicted Median House Value: ${predicted_value[0]:,.2f}")


# Visualize the relationship between a feature and the target variable
plt.scatter(X['MedInc'], y, alpha=0.5)
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value')
plt.title('Median House Value vs. Median Income')
plt.show()


