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

        # recording what actions were taken 
        self.action_list = [] 

        self.num_states = num_states
        self.num_actions = num_actions

        
        self.Q = np.zeros([self.num_states, self.num_actions])

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
            # if the random number is less than epsilon it will do a random action (Exploring)
        else: 
            action = np.argmax(self.Q[state, :])
            self.action_list.append(action) # Record the actions taken
            return action



    def update(self, state, action, reward, next_state, done):

        # by the book  original thinking
        """
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + (self.gamma * np.argmax(self.Q[next_state, action])) - self.Q[state, action])
        state = next_state 

        return state, self.Q
        """

        # actual working function 

        current = self.Q[state, action] # Getting the current state and action 

        if done: 
            target = reward # if they hit the target? getting through the pipes 
        else:
            best_next = np.max(self.Q[next_state, : ]) # Getting next best action to use 
            target = reward + self.gamma * best_next # next best reward when just playing 

        self.Q[state, action] = current + self.alpha * (target - current) # setting the next state. Based upon the rewards and actions on the previous state


    def decay_epsilon(self): # Reduces exploration after each episode 
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay) # if it reaches min_epsilon it won't decay further
        return self.epsilon

