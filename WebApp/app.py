from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
from PIL import Image
from watchdog.events import EVENT_TYPE_OPENED

app = Flask(__name__)

# load pre-trained model
with open('D:/Neural_Projects/Computational-Intelligence-and-Statistical-Learning-master/1st_project (SVM)/Iris/svm_model.pkl','rb') as f:
    clf = pickle.load(f)

# x_test_data1 = [5.1,3.5,1.4,0.2]
# x_test_data3 = [6.8, 2.8,4.8,1.4]


@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/IrisText', methods = ['GET'])
def IrisText():
    return render_template('IrisText.html')

@app.route('/predictText', methods =['POST'])
def predictText():
        # Get the input data from the request
       # Extract the input data from the HTML form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        input_data = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)
        # Use the trained SVM model to predict the class of the input data
        prediction = clf.predict(input_data)
        return f'The predicted class is: {int(prediction[0])}'


@app.route('/IrisImage', methods = ['GET'])
def IrisImage():
    return render_template('IrisImage.html')


@app.route('/predictImage', methods=['POST'])
def predict():
    # Load the image from the request
    file = request.files['file']
    image = Image.open(file.stream)

    image2 = image.resize((150, 150))
    image3 = image2.convert('L')
    image_array1 = np.array(image3)
    image_array2 = image_array1.flatten()
    image_array3 = image_array2 / 255.0
    image_array4 = image_array3.reshape(1, -1)
    # Use the pre-trained machine learning model to predict the species of the iris plant in the image
    prediction = clf.predict(image_array4)
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Return the predicted species as a JSON response
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return jsonify({'prediction': species[predicted_class]})
     



if __name__ == '__main__':
    # Start the Flask appimg
    app.run(debug=True)



