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


# # Tabular Model Free Algorithms

# ### Utility functions

# In[3]:


def initialize_coeff(env, max_episodes, eta, epsilon, seed):
    """
    Define random seed, eta (decaying learning rate), 
    epsilon (decaying exploration rate) and Q values (for state action pairs)
    """
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes) 
    epsilon = np.linspace(epsilon, 0, max_episodes) 
    q = np.zeros((env.n_states, env.n_actions))
    
    return q, epsilon, eta, random_state


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


# # SARSA

# In[4]:


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Initialize coefficients, Q values and random state.
    Iterating through max episodes, start game and choose an action.
    Until game over, play action to get rewards and new state. Use new state to select new action.
    Update Q values based on states, actions and coefficients.
    When all iterations are over, return optimal values and policy.
    """
    q, epsilon, eta, random_state = initialize_coeff(env, max_episodes, eta, epsilon, seed)
    
    for i in range(max_episodes):   
        s, done = env.reset(), False          
        a = select_action(env, q[s], random_state, epsilon, i)
        
        while(not done):
            s_new, r, done = env.step(a)
            a_new = select_action(env, q[s_new], random_state, epsilon, i)
            q[s,a] += eta[i] * (r + gamma * q[s_new, a_new] - q[s,a])
            s, a = s_new, a_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


# # Q-Learning

# In[5]:


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Initialize coefficients, Q values and random state.
    Iterating through max episodes: start game.
    Until game over, choose action, play action to get rewards and new state. 
    Use new state to select best new action.
    Update Q values based on states, actions and coefficients.
    When all iterations are over, return optimal values and policy.
    """
    q, epsilon, eta, random_state = initialize_coeff(env, max_episodes, eta, epsilon, seed)

    for i in range(max_episodes):
        s, done = env.reset(), False
        
        while(not done):
            a = select_action(env, q[s], random_state, epsilon, i)
            s_new, r, done = env.step(a)
            q_max = max(q[s_new])
            q[s,a] += eta[i] * (r + gamma * q_max - q[s,a])
            s = s_new
            
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


# ## Config

# In[13]:


max_episodes = 20000
eta = 0.5
epsilon = 0.5
gamma = 0.9





