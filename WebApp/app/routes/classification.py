from flask import Blueprint, request, render_template, current_app
import numpy as np

classification_bp = Blueprint('classification', __name__)

@classification_bp.route('/Classification')
def classification():
    return render_template('Classification.html')
