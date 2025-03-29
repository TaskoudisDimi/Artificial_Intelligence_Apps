from flask import Blueprint, render_template, request, jsonify
from models.Translate.Transformer import translate  # Assuming you have a translate function in your Transformer model

language_technology_bp = Blueprint('language_technology', __name__)

@language_technology_bp.route('/language_technology', methods=['GET'])
def language_technology():
    return render_template('LanguageTechnology.html')

@language_technology_bp.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        english_text = request.form['english_text']
        translated_text = translate(english_text)  # Implement the translation function in your model
        return jsonify({'translation': translated_text})
    return jsonify({'error': 'Invalid request'}), 400