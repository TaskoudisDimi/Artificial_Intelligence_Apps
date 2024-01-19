from flask import render_template, Flask, request, jsonify
import torch
import pandas as pd
from model import TripModel  # Replace with your actual model class
import numpy as np




app = Flask(__name__)


input_size = 4
hidden_size = 128  
num_layers = 2
output_size = 1

# Load the trained PyTorch model
model = TripModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/TrainedModels/Regression/Trips/trips_model.pth"))
model.eval() 


def preprocess_data(data):
    # Implement your preprocessing logic here
    return np.array(data)

@app.route('/predictPickup', methods=['POST'])
def predict_pickup():
    if request.method == 'POST':
        # Extract input parameters from the form
        start_postal_code = float(request.form['Start_PostalCode'])
        day = float(request.form['Day'])
        month = float(request.form['Month'])
        year = float(request.form['Year'])

        # Preprocess the input data
        input_data = preprocess_data([start_postal_code, day, month, year])

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(1)

        # Make predictions
        with torch.no_grad():
            prediction = model(input_tensor)

        # Convert prediction to a standard Python type (e.g., float)
        predicted_value = prediction.item()

        return render_template('result.html', prediction=predicted_value)
    else:
        return "Invalid request method"

@app.route('/')
def test():
    return render_template('testAPI.html')

if __name__ == '__main__':
    app.run(debug=True)