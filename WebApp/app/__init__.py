from flask import Flask
from .routes import routes_bp
from .routes.classificationIris import classification_bp 
from .routes.clustering import clustering_bp
from app.helpers.loader import load_models_from_cloud  # Import the loader

def create_app():
    app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

    app.config.from_object('config.Config')

    # Load models from cloud and store in app context
    app.models = load_models_from_cloud(app.config)

    app.register_blueprint(routes_bp)
    app.register_blueprint(classification_bp)  # Register classification routes
    app.register_blueprint(clustering_bp)  # Register clustering routes

    return app
