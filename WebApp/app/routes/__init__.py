from flask import Blueprint, render_template

# Initialize the routes blueprint
routes_bp = Blueprint('routes', __name__)

# Home route
@routes_bp.route('/')
def home():
    return render_template('Home.html')  # Ensure the file is named correctly in templates folder

@routes_bp.route('/classification')
def classification():
    return render_template('Classification.html')

@routes_bp.route('/classificationIris')
def classificationIris():
    return render_template('Classification_Iris.html')

@routes_bp.route('/clustering')
def clustering():
    return render_template('Clustering.html')

@routes_bp.route('/clusteringIris')
def clusteringIris():
    return render_template('Clustering_Iris.html')

@routes_bp.route('/regression')
def regression():
    return render_template('Regression.html')

@routes_bp.route('/computer_vision')
def computer_vision():
    return render_template('ComputerVision.html')

@routes_bp.route('/activity_recognition')
def activity_recognition():
    return render_template('Activity_Recognition.html')

@routes_bp.route('/language_technology')
def language_technology():
    return render_template('LanguageTechnology.html')

@routes_bp.route('/chatbot')
def chatbot():
    return render_template('Chatbot.html')

@routes_bp.route('/about')
def about():
    return render_template('About.html')

@routes_bp.route('/contact')
def contact():
    return render_template('Contact.html')