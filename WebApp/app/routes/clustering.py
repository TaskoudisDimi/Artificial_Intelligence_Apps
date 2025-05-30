from flask import Blueprint, render_template, request, jsonify, current_app
import numpy as np

clustering_bp = Blueprint('clustering', __name__)

@clustering_bp.route('/Clustering')
def clustering():
    return render_template('Clustering.html')

@clustering_bp.route('/Clustering_Iris', methods=['GET', 'POST'])
def cluster_iris():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form.get('sepal_length'))
            sepal_width = float(request.form.get('sepal_width'))
            if not all([sepal_length, sepal_width]):
                message = "Please set all the features"
                return render_template('Clustering_Iris.html', message=message)
            
            input_data = np.array([[sepal_length, sepal_width]])
            
            models = current_app.models
            if not models:
                message = "No models loaded. Please check the server configuration."
                return render_template('Clustering_Iris.html', message=message)
            prediction = models['KMeans_Iris_model'].predict(input_data)
            result = ["Setosa", "Versicolor", "Virginica"][int(prediction[0])] if prediction is not None else "Unknown"
            return render_template('Clustering_Iris.html', result=result)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid input. Please enter valid numbers.'})

    return render_template('Clustering_Iris.html')

@clustering_bp.route('/Clustering_BreastCancer', methods=['GET', 'POST'])
def clustering_breast_cancer_predict():
    # if request.method == 'POST':
    #     try:
    #         mean_radius = float(request.form.get('mean_radius'))
    #         mean_texture = float(request.form.get('mean_texture'))

    #         user_input = np.array([[mean_radius, mean_texture]])
    #         user_input = scaler.transform(user_input)

    #         models = current_app.models
    #         if not models:
    #             message = "No models loaded. Please check the server configuration."
    #             return render_template('Clustering_BreastCancer.html', message=message)
           
    #         predicted_cluster = KMeans_breast_cancer_model.predict(user_input)
    #         if predicted_cluster is None:
    #             return render_template('Clustering_BreastCancer.html')
    #         else:
    #             result = int(predicted_cluster[0])
    #             result = "Setosa" if result == 0 else "Versicolor" if result == 1 else "Virginica"
    #             return render_template('Clustering_BreastCancer.html', result=result)

    #     except (ValueError, TypeError):
    #         return jsonify({'error': 'Invalid input. Please enter valid numbers.'})

    return render_template('Clustering_BreastCancer.html')