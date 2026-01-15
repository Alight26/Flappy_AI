import numpy as np
import random 
from env_test import env

class Agent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay, min_epsilon, num_states, num_actions, num_episode):
        
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount Factor
        self.epsilon = epsilon # Exploration rate 
        self.epsilon_decay = epsilon_decay # multiply to epsilon to decay it 
        self.min_epsilon = min_epsilon # min epsilon so it will still explore 
        self.num_episode = num_episode # number of episodes 
        

        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros(self.num_states, self.num_actions)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
            # if the random number is less than epsilon it will do a random action (Exploring)
        else: 
            return np.argmax(self.Q[state, :])
        



    def update(self, state, action, reward, next_state, done):
        pass 

    def decay_epsilon(self): # Reduces exploration after each episode 
        pass

