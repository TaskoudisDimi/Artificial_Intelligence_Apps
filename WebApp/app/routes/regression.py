from flask import Blueprint, request, render_template, jsonify, current_app
import numpy as np


regression_bp = Blueprint('regression', __name__)

@regression_bp.route('/Regression', methods=['GET'])
def regression():
    return render_template('Regression.html')

@regression_bp.route('/RegressionIris', methods=['POST'])
def regression_Iris():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form.get('sepal_length'))
            sepal_width = float(request.form.get('sepal_width'))
            petal_length = float(request.form.get('petal_length'))
            petal_width = float(request.form.get('petal_width'))

            user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            # Get models from app context
            models = current_app.models  
            # Check if models are loaded
            if not models:
                message = "No models loaded. Please check the server configuration."
                return render_template('Regression_Iris.html', message=message)
            predicted_value = models["Regression_Iris_model"].predict(user_input)

            return render_template('Regression_Iris.html', predicted_value=predicted_value[0])

        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid input. Please enter valid numbers.'})

    return render_template('Regression_Iris.html', predicted_value=None)
@regression_bp.route('/RegressionHouse', methods=['POST'])
def regression_House():
    try:
        # Fetch 8 numerical features
        numerical_inputs = [
            float(request.form.get('med_inc')),
            float(request.form.get('house_age')),
            float(request.form.get('ave_rooms')),
            float(request.form.get('ave_bedrms')),
            float(request.form.get('population')),
            float(request.form.get('ave_occup')),
            float(request.form.get('latitude')),
            float(request.form.get('longitude'))
        ]
        # Fetch categorical feature and create binary feature
        ocean_proximity = request.form.get('ocean_proximity')
        is_inland = 1.0 if ocean_proximity == 'INLAND' else 0.0

        # Combine inputs
        inputs = numerical_inputs + [is_inland]
        inputs_2d = np.array(inputs).reshape(1, -1)

        # Check input shape
        if inputs_2d.shape[1] != 9:
            raise ValueError(f"Expected 9 features, but got {inputs_2d.shape[1]}")

        models = current_app.models
        if not models:
            message = "No models loaded. Please check the server configuration."
            return render_template('Regression_House.html', message=message)

        predicted_value = models["regression_house"].predict(inputs_2d)
        predicted_value = f"${predicted_value[0] * 100000:.2f}"  # Convert to dollars

        return render_template('Regression_House.html', predicted_value=predicted_value)

    except Exception as e:
        return render_template('Regression_House.html', error="Error: " + str(e))