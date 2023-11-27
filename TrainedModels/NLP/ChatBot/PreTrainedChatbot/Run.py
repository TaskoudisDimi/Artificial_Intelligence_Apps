import numpy as np
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from Model import ChatBot
    
    
# build a ChatBot object
bot = ChatBot()
# start chatting
while True:
    # receive user input
    bot.user_input()
    # check whether to end chat
    if bot.end_chat:
        break
    # output bot response
    bot.bot_response()   
    
    
    
    
    
    
    
    