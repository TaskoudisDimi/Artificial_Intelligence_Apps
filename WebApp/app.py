from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
from watchdog.events import EVENT_TYPE_OPENED
import cv2 as cv
import joblib


app = Flask(__name__, static_url_path='/static', static_folder='static')

# Model based on Text classification
# load pre-trained model
# with open('D:/Neural_Projects/Computational-Intelligence-and-Statistical-Learning-master/1st_project (SVM)/Iris/svm_model.pkl','rb') as f:
#     clf = pickle.load(f)

# with open('C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/WebApp/Models/svm_model_iris.pkl','rb') as f:
#     clf = pickle.load(f)

# Load the SVM model from the file
model_filename = 'C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/WebApp/Models/svm_model_iris.pkl'
SVM_model = joblib.load(model_filename)
# Load the KNN model from the file
model_filename = 'C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/WebApp/Models/KNN_model_iris.pkl'
KNN_model = joblib.load(model_filename)
# Load the KNearestCentroid model from the file
model_filename = 'C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/WebApp/Models/Nearest_model_iris.pkl'
KNearestCentroid_model = joblib.load(model_filename)



@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/Classification')
def Classification():
    return render_template('Classification.html')

@app.route('/Classification_Iris', methods=['GET'])
def Classification_Iris():
    return render_template('Classification_Iris.html')

@app.route('/ClassificationIris', methods=['POST'])
def ClassificationIris():
    try:
        # Extract the input data from the HTML form
        SepalLength = request.form.get('SepalLength')
        SepalWidth = request.form.get('SepalWidth')
        PetalLength = request.form.get('PetalLength')
        PetalWidth = request.form.get('PetalWidth')

        # Check if any of the features are null or empty strings
        if not all([SepalLength, SepalWidth, PetalLength, PetalWidth]):
            message = "Please set all the features"
            return render_template('Classification_Iris.html', message=message)
        
        # Convert features to floats if they are not null or empty strings
        SepalLength = float(SepalLength)
        SepalWidth = float(SepalWidth)
        PetalLength = float(PetalLength)
        PetalWidth = float(PetalWidth)
        
        input_data = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(1, -1)

        if 'SVM' in request.form:
            prediction = SVM_model.predict(input_data)
        elif 'KNN' in request.form:
            prediction = SVM_model.predict(input_data)
        elif 'KNearestCentroid' in request.form:
            prediction = SVM_model.predict(input_data)

        if(prediction is None):
            return render_template('Classification_Iris.html')
        else:
            result = int(prediction[0])
            if(result == 0):
                result = "Setosa"
            elif result == 1:
                result = "Versicolor"
            else:
                result = "Virginica"
            return render_template('Classification_Iris.html', result=result)
        
        
    except (ValueError, TypeError) as e:
        message = "Invalid input format. Please provide valid numeric values for all features."
        return render_template('Classification_Iris.html', message=message)


@app.route('/Classification_BreastCancer')
def Classification_BreastCancer():
    return render_template('Classification_BreastCancer.html')





@app.route('/Classification_Mnist')
def Classification_Mnist():
    return render_template('Classification_Mnist.html')











@app.route('/Regression')
def Regression():
    return render_template('Regression.html')


@app.route('/Clustering')
def Clustering():
    return render_template('Clustering.html')








# @app.route('/predictIrisSVM', methods=['POST'])
# def predictIrisSVM():
#     try:
#         # Extract the input data from the HTML form
#         feature1 = request.form.get('feature1')
#         feature2 = request.form.get('feature2')
#         feature3 = request.form.get('feature3')
#         feature4 = request.form.get('feature4')

#         # Check if any of the features are null or empty strings
#         if not all([feature1, feature2, feature3, feature4]):
#             message = "Please set all the features"
#             return render_template('SVM_Iris_Classification.html', message=message)
        
#         # Convert features to floats if they are not null or empty strings
#         feature1 = float(feature1)
#         feature2 = float(feature2)
#         feature3 = float(feature3)
#         feature4 = float(feature4)
        
#         input_data = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
#         prediction = svm_model.predict(input_data)
#         return render_template('Results.html', result=int(prediction[0]))
        
#     except (ValueError, TypeError) as e:
#         message = "Invalid input format. Please provide valid numeric values for all features."
#         return render_template('SVM_Iris_Classification.html', message=message)
       
        
# @app.route('/IrisImage', methods = ['GET'])
# def IrisImage():
#     return render_template('IrisImage.html')

# @app.route('/predictImage', methods=['POST'])
# def predict():
#     # Load the image from the request
#     file = request.files['file']
#     image = Image.open(file.stream)

#     image2 = image.resize((150, 150))
#     image3 = image2.convert('L')
#     image_array1 = np.array(image3)
#     image_array2 = image_array1.flatten()
#     image_array3 = image_array2 / 255.0
#     image_array4 = image_array3.reshape(1, -1)
#     # Use the pre-trained machine learning model to predict the species of the iris plant in the image
#     prediction = clf.predict(image_array4)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     # Return the predicted species as a JSON response
#     species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
#     return jsonify({'prediction': species[predicted_class]})
     


# @app.route('/predit_CNN_Mnist_Image', methods = ['GET'])
# def IrisImage():
#     path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSs6m6DyeI4XQcfZb28KlYj_rs0Ru0Raq-3_lS7ULqrv1zR9cuUAv1tAzSjFGS9ORcKFCc&usqp=CAU"
#     r = requests.get(path)
#     with BytesIO(r.content) as f:
#        img = Image.open(f).convert(mode="L")
#        img = img.resize((28, 28))
#     x = (255 - np.expand_dims(np.array(img), -1))/255.

#     pred = inference(path, lenet, device=device)
#     pred_idx = np.argmax(pred)
#     print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx]*100} %")

#     return render_template('IrisImage.html')





if __name__ == '__main__':
    # Start the Flask appimg
    app.run(debug=True)



