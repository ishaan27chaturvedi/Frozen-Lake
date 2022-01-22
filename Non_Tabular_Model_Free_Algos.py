#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from itertools import product
import contextlib
from Environment import * 


# In[2]:


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


# # Non Tabular Model Free Algorithms

# ## Linear 

# In[3]:


class LinearWrapper:
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        self.absorbing_state = self.env.absorbing_state

    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


# ### Utility functions

# In[11]:


def initialize_coeff(env, max_episodes, eta, epsilon, seed):
    """
    Define random seed, eta (decaying learning rate), 
    epsilon (decaying exploration rate) and Q values (for state action pairs)
    """
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes) 
    epsilon = np.linspace(epsilon, 0, max_episodes) 
    theta = np.zeros(env.n_features) 
    
    return theta, epsilon, eta, random_state


def randomBestAction(random_state, mean_rewards):
    """
    Get an array of best actions based on Q values (mean_rewards)
    Break ties randomly and return one of the best actions
    """
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    
    return random_state.choice(best_actions, 1)[0]  


def select_action(env, q, random_state, epsilon, i):
    """
    Select action a for state s according to an e-greedy policy based on Q values.
    Use Epsilon Greedy method to decide whether to take best action or random action.
    """
    
    if(random_state.random(1) < epsilon[i]):
        a = random_state.choice(range(env.n_actions)) 
    else:
        a = randomBestAction(random_state, q)
    
    return a


# ## Linear SARSA

# In[5]:


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Initialize coefficients, theta values and random state.
    Iterating through max episodes, start game, define q and choose an action.
    Until game over, play action to get rewards and new features. 
    Use new features to update new q values and theta values. Then select new action.
    Update Q values, delta and theta based on features, rewards and coefficients.
    When all iterations are over, return optimal values and policy.
    """
    theta, epsilon, eta, random_state = initialize_coeff(env, max_episodes, eta, epsilon, seed)
    
    for i in range(max_episodes):
        features, done = env.reset(), False
        q = features.dot(theta)
        a = select_action(env, q, random_state, epsilon, i)

        while not done:
            features_prime, r, done = env.step(a)
            delta = r - q[a]
            q = features_prime.dot(theta)  
            a_prime = select_action(env, q, random_state, epsilon, i)

            delta += (gamma * q[a_prime])
            theta += eta[i] * delta * features[a]
            features = features_prime
            a = a_prime

    return theta


# ## Linear Q-Learning

# In[6]:


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Initialize coefficients, Q values and random state.
    Iterating through max episodes: start game.
    Until game over, choose action, play action to get rewards and new features. 
    Use new features to select best new action.
    Update Q values based on features, theta, actions and coefficients.
    When all iterations are over, return optimal values and policy.
    """
    theta, epsilon, eta, random_state = initialize_coeff(env, max_episodes, eta, epsilon, seed)

    for i in range(max_episodes):
        features, done = env.reset(), False
        q = features.dot(theta)

        while not done:
            a = select_action(env, q, random_state, epsilon, i)
            features_prime, r, done = env.step(a)
            delta = r - q[a]
            q = features_prime.dot(theta)
            
            delta += (gamma * max(q))
            theta += eta[i] * delta * features[a]
            features = features_prime

    return theta


# ### Config

# In[9]:


linear_env = LinearWrapper(env)

max_episodes = 20000
eta = 0.5
epsilon = 0.5
gamma = 0.9





