from flask import render_template, Flask, request, jsonify
import torch
import pandas as pd
from model import UpdatedLSTMModel  # Replace with your actual model class
import numpy as np




app = Flask(__name__)

# Load the trained PyTorch model
model = UpdatedLSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load("C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/TrainedModels/Regression/Trips/trips_model.pth"))
model.eval() 

# Assuming you have a StandardScaler object saved during training
scaler = StandardScaler()  # Replace with your actual scaler
# Load the scaler parameters (mean and std) saved during training
scaler.mean_ = np.array([mean_start_postal_code, mean_day, mean_month, mean_year])
scaler.scale_ = np.array([std_start_postal_code, std_day, std_month, std_year])

def preprocess_data(data):
    # Implement your preprocessing logic here
    return scaler.transform(data.reshape(1, -1))

@app.route('/predictPickup', methods=['POST'])
def predict_pickup():
    if request.method == 'POST':
        # Extract input parameters from the form
        start_postal_code = float(request.form['Start_PostalCode'])
        day = float(request.form['Day'])
        month = float(request.form['Month'])
        year = float(request.form['Year'])

        # Preprocess the input data
        input_data = preprocess_data(np.array([start_postal_code, day, month, year]))

        # Convert to PyTorch tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            prediction = model(input_tensor.unsqueeze(1))

        # Convert prediction to a standard Python type (e.g., float)
        predicted_value = prediction.item()

        return render_template('result.html', prediction=predicted_value)
    else:
        return "Invalid request method"

@app.route('/')
def test():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)
