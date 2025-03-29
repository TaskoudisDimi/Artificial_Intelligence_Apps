# Flask AI Application

This project is a Flask-based web application that provides various AI functionalities, including classification, clustering, regression, computer vision, chatbot interactions, activity recognition, and language translation.

## Project Structure

```
flask-ai-app
├── app
│   ├── __init__.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── clustering.py
│   │   ├── regression.py
│   │   ├── computer_vision.py
│   │   ├── chatbot.py
│   │   ├── activity_recognition.py
│   │   └── language_technology.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   ├── cifar.py
│   │   ├── iris.py
│   │   ├── breast_cancer.py
│   │   └── utils.py
│   ├── templates
│   │   ├── Home.html
│   │   ├── Classification.html
│   │   ├── Classification_Iris.html
│   │   ├── Clustering.html
│   │   ├── Clustering_Iris.html
│   │   ├── Clustering_BreastCancer.html
│   │   ├── Regression.html
│   │   ├── Regression_Iris.html
│   │   ├── Regression_House.html
│   │   ├── ComputerVision.html
│   │   ├── ComputerVision_MNIST.html
│   │   ├── ComputerVision_MNIST_Up_image.html
│   │   ├── ComputerVision_CIFAR10.html
│   │   ├── RealTimeFaceDetection.html
│   │   ├── Activity_Recognition.html
│   │   ├── LanguageTechnology.html
│   │   ├── English_To_German.html
│   │   ├── Chatbot.html
│   │   ├── PretrainedChatbot.html
│   │   ├── CustomChatbot.html
│   │   ├── FakeImage.html
│   │   └── FromTextToImage.html
│   └── static
│       ├── css
│       ├── js
│       └── images
├── models
│   ├── Mnist
│   │   └── model.py
│   ├── Cifar
│   │   └── Net.py
│   ├── Chatbot
│   │   ├── PretrainedChatbot
│   │   │   └── Model.py
│   │   └── CustomChatbot
│   │       └── model.py
│   └── Translate
│       └── Transformer.py
├── models
│   └── (downloaded model files)
├── requirements.txt
├── config.py
├── run.py
└── README.md
```

## Features

- **Classification**: Classify iris flowers based on their features.
- **Clustering**: Perform clustering on iris and breast cancer datasets.
- **Regression**: Predict values based on regression models for iris and house prices.
- **Computer Vision**: Recognize digits using MNIST and classify images using CIFAR-10.
- **Chatbot**: Interact with both pretrained and custom chatbots.
- **Activity Recognition**: Recognize activities from video inputs.
- **Language Technology**: Translate text from English to German.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-ai-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run.py
   ```

## Usage

- Access the application in your web browser at `http://127.0.0.1:5000/`.
- Navigate through the various functionalities using the provided links.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.