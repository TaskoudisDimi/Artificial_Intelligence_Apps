from flask import Blueprint, render_template, request, jsonify
from Old_Models.Chatbot.CustomChatbot.model import response

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chatbot', methods=['GET'])
def chat_bot():
    return render_template('Chatbot.html')

@chatbot_bp.route('/custom_chat', methods=['POST'])
def custom_chat():
    user_input = request.form['user_input']
    if user_input.lower().strip() in ['bye', 'quit', 'exit']:
        return "ChatBot: See you soon! Bye!"
    bot_response = response(user_input)
    return "ChatBot: " + bot_response