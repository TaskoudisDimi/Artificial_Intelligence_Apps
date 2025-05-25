import os
from flask import Blueprint, render_template, request, jsonify, session, current_app
from transformers import AutoTokenizer, pipeline, Conversation
import tensorflow as tf
import bleach
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chatbot', methods=['GET'])
def chat_bot():
    return render_template('Chatbot.html')


@chatbot_bp.route("/predict", methods=["POST"])
def chat_bot_response():
    msg = request.form["message"]  # match the 'name' of the input field in HTML
    return jsonify({"bot_response": get_Chat_response(msg)})


def get_Chat_response(text):
    # Encode user input
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Generate a response without chat history
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the bot's response
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response
