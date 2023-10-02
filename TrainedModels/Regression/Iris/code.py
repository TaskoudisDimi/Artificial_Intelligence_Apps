# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Evaluate the model
score = regression_model.score(X_test, y_test)
print(f"Model Accuracy: {score:.2f}")

# Save the trained Linear Regression model to a file
model_filename = 'iris_regression_model.pkl'
joblib.dump(regression_model, model_filename)