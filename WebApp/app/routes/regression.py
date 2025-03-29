from flask import Blueprint, request, render_template, jsonify
import numpy as np
from app.models.iris import RegressionIrisModel
from app.models.breast_cancer import RegressionHouseModel

regression_bp = Blueprint('regression', __name__)

@regression_bp.route('/Regression', methods=['GET'])
def regression():
    return render_template('Regression.html')

@regression_bp.route('/Regression_iris', methods=['POST'])
def regression_iris():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form.get('sepal_length'))
            sepal_width = float(request.form.get('sepal_width'))
            petal_length = float(request.form.get('petal_length'))
            petal_width = float(request.form.get('petal_width'))

            user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            predicted_value = RegressionIrisModel.predict(user_input)

            return render_template('Regression_Iris.html', predicted_value=predicted_value[0])

        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid input. Please enter valid numbers.'})

    return render_template('Regression_Iris.html', predicted_value=None)

@regression_bp.route('/Regression_house', methods=['POST'])
def regression_house():
    try:
        inputs = [float(x) for x in request.form.values()]
        prediction = RegressionHouseModel.predict([inputs])[0]
        predicted_value = f"${prediction:.4f}"
        
        return render_template('Regression_House.html', prediction=predicted_value)
    
    except Exception as e:
        return render_template('Regression_House.html', error="Error: " + str(e))