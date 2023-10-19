import numpy as np
import random
import torch
import numpy as np
from SnakePytorchReinforcement import SnakeGameAI
from collection import deque



# deque store memory



MAX_MEMORY = 100_00
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # TODO: model, trainer
        
        
        
        
    def get_state(self):
        pass
    
    def remember(self, action, reward, next_state, done):
        pass
    
    
    def train_long_memory(self):
        pass
        
    def train_short_memory(self):
        pass
        
    
    def get_action(self, state):
        pass
        
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = []
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get(game)
        
        #train short memory
        


if __name__ == "__mina__":
    train()


























