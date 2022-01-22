#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import random
from itertools import product
import contextlib
from Environment import * 


# ### Defining Print Options

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


# # Tabular Model Based Algorithms

# ## Bellman Equation 

# In[3]:


def calcluate_value(env, gamma, values, s, a):
    """
    V_pi(s) = sum ( Pass' * [ Rass' + gamma*V_pi(s') ] )
    """
    value = sum(
                [ env.p(next_s, s, a) * 
                ( env.r(next_s, s, a) + gamma * values[next_s] )
                for next_s in range(env.n_states) ]
                ) 
    
    return value


# # Policy Iteration

# In[19]:


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    """
    Update value of each state, until max_iterations or delta < theta
    """
    values = np.zeros(env.n_states, dtype=np.float)
    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = values[s]
            values[s] = calcluate_value(env, gamma, values, s, policy[s])
            delta = max(delta, abs(v - values[s]))
        if delta < theta:   
            break
            
    return values


def policy_improvement(env, values, gamma):
    """
    Updates policy: for each state, replace current action with the highest value rewarding action
    """
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax( [ calcluate_value(env, gamma, values, s, a) for a in range(env.n_actions) ] )

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    """
    Create a random policy. Alternatively evaluate and improve the policy until no improvement can be made.
    """
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
        
    policy_initial = None
    
    while(np.array_equal(policy_initial, policy)==False):
        policy_initial = policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, value, gamma)
        
    return policy, value


# ### Run Policy iteration

# In[21]:


gamma = 0.9
theta = 0.001
max_iterations = 100000



# # Value Iteration

# In[22]:


def value_iteration(env, gamma, theta, max_iterations, values=None):
    """
    Start with random values and iterate until optimal values have been found.
    Calculate policy in one sweep based on optimal values.
    """
    if values is None:
        values = np.zeros(env.n_states)
    else:
        values = np.array(values, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = values[s]
            values[s] = max( [ calcluate_value(env, gamma, values, s, a) for a in range(env.n_actions) ] )
            delta = max(delta, abs(v - values[s]))
        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax( [ calcluate_value(env, gamma, values, s, a) for a in range(env.n_actions) ] )

    return policy, values





