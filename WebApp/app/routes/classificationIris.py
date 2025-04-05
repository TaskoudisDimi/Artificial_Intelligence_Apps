from flask import Blueprint, request, render_template, current_app
import numpy as np

classification_bp = Blueprint('classification', __name__)

@classification_bp.route('/ClassificationIris', methods=['POST'])
def classify_iris():
    try:
        SepalLength = request.form.get('SepalLength')
        SepalWidth = request.form.get('SepalWidth')
        PetalLength = request.form.get('PetalLength')
        PetalWidth = request.form.get('PetalWidth')

        if not all([SepalLength, SepalWidth, PetalLength, PetalWidth]):
            message = "Please set all the features"
            return render_template('Classification_Iris.html', message=message)

        input_data = np.array([float(SepalLength), float(SepalWidth), float(PetalLength), float(PetalWidth)]).reshape(1, -1)
        print(f"Input data: {input_data}")
        # Get models from app context
        models = current_app.models  
        print(f"Models loaded: {models.keys()}")
        # Check if models are loaded
        if not models:
            message = "No models loaded. Please check the server configuration."
            return render_template('Classification_Iris.html', message=message)
        # Choose model based on form input  
        print(f"Form data: {request.form}")
        if 'SVM' in request.form:
            prediction = models["SVM_Iris_model"].predict(input_data)
        elif 'KNN' in request.form:
            prediction = models["KNN_Iris_model"].predict(input_data)
        elif 'KNearestCentroid' in request.form:
            prediction = models["KNearestCentroid_Iris_model"].predict(input_data)
        else:
            message = "Invalid model selection."
            return render_template('Classification_Iris.html', message=message)

        # Convert prediction result
        result = ["Setosa", "Versicolor", "Virginica"][int(prediction[0])] if prediction is not None else "Unknown"
        return render_template('Classification_Iris.html', result=result)

    except (ValueError, TypeError) as e:
        message = "Invalid input format. Please provide valid numeric values for all features."
        return render_template('Classification_Iris.html', message=message)
