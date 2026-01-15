import numpy as np

class Agent:
    def __init__(self, alpha, gamma, epsilon, num_states, num_actions):
        self.Q = np.array([self.num_states], [self.num_actions])
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount Factor
        self.epsilon = epsilon # Exploration rate 

        self.num_states = num_states
        self.num_actions = num_actions

    def choose_action(self, state):
        pass

    def update(self, state, action, reward, next_state, done):
        pass 

    def decay_epsilon(self): # Reduces exploration after each episode 
        pass
