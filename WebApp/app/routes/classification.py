from flask import Blueprint, request, render_template, jsonify
import numpy as np
from app.models.iris import SVM_Iris_model, KNN_Iris_model, KNearestCentroid_Iris_model
from app.models.breast_cancer import KMeans_breast_cancer_model
from app.models.utils import scaler
from app.models.regression import Regression_Iris_model, Regression_House_model

classification_bp = Blueprint('classification', __name__)

@classification_bp.route('/Classification')
def classification():
    return render_template('Classification.html')

@classification_bp.route('/Classification_Iris', methods=['GET'])
def classification_iris():
    return render_template('Classification_Iris.html')

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
        
        SepalLength = float(SepalLength)
        SepalWidth = float(SepalWidth)
        PetalLength = float(PetalLength)
        PetalWidth = float(PetalWidth)
        
        input_data = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(1, -1)

        if 'SVM' in request.form:
            prediction = SVM_Iris_model.predict(input_data)
        elif 'KNN' in request.form:
            prediction = KNN_Iris_model.predict(input_data)
        elif 'KNearestCentroid' in request.form:
            prediction = KNearestCentroid_Iris_model.predict(input_data)

        if prediction is None:
            return render_template('Classification_Iris.html')
        else:
            result = int(prediction[0])
            if result == 0:
                result = "Setosa"
            elif result == 1:
                result = "Versicolor"
            else:
                result = "Virginica"
            return render_template('Classification_Iris.html', result=result)
        
    except (ValueError, TypeError) as e:
        message = "Invalid input format. Please provide valid numeric values for all features."
        return render_template('Classification_Iris.html', message=message)