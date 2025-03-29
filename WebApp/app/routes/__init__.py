from flask import Blueprint

# Initialize the routes blueprint
routes_bp = Blueprint('routes', __name__)

# Import all route modules to register them with the blueprint
from . import classification
from . import clustering
from . import regression
from . import computer_vision
from . import chatbot
from . import activity_recognition
from . import language_technology