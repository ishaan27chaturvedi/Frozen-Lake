#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from itertools import product
import contextlib


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


# # Enviornment Model Class

# In[4]:


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


# # Environment Class

# In[5]:


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


# # Frozen Lake Environment Class

# In[6]:


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip  # parameterizable slip probability, initially 0.1

        n_states = self.lake.size + 1  # + 1 to include the absorption state
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)  # initial p
        pi[np.where(self.lake_flat == '&')[0]] = 1.0  # setting start state p to 1

        self.absorbing_state = n_states - 1   # set to 16 - outside lake_flat, ranging from 0-15
        print(self.absorbing_state)
        # Initializing environment
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # Up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        #up, left, down, and right.
        #self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.transition_probas = np.zeros((self.n_states, self.n_states, self.n_actions))
        #Iterate over all combinations
        self.all_states = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        for state_index in range(n_states):
            for other_state_index in range(n_states):
                for action_index, action in enumerate(self.actions):
                
                    if state_index != self.absorbing_state:
                        character = self.lake_flat[state_index]
                        
                        if character == '$' or character == '#':
                            if other_state_index == self.absorbing_state:
                                self.transition_probas[other_state_index, state_index, action_index]=1.0
                                    
                        elif character != '$' and character != '#':         
        
                            if other_state_index != self.absorbing_state:                             
                                
                                state = self.all_states[state_index]
                                other_state = self.all_states[other_state_index]
                                if self.apply_action(state,action) == other_state:
                                    self.transition_probas[other_state_index, state_index, action_index]=1.0 - self.slip
                                

                                self.transition_probas[other_state_index, state_index, action_index]+=                                 self.slip_multiplyer(other_state,state)*(self.slip/n_actions)
    
                    if state_index == self.absorbing_state and other_state_index == self.absorbing_state:
                            self.transition_probas[other_state_index, state_index, action_index]=1.0

                
    def apply_action(self,state,action):
        new_state = (state[0]+action[0],state[1]+action[1])
        if self.valid_state(new_state):
            return new_state
        return state

    
    def valid_state(self, state):
        if state[0] >= 0 and state[0] <self.lake.shape[0] and state[1] >= 0 and state[1] <self.lake.shape[1]:
            return True
        return False
    

    def slip_multiplyer(self,state,goal_state):
        count = 0
        for action in self.actions:
            if self.apply_action(state,action) == goal_state:
                count+=1
        return count
    
    def get_index(self, state):
        return state[0]*self.lake.shape[0] + state[1]
    
    
    def step(self, action):
        state, reward, done = Environment.step(self, action)  # else, transition normally
        done = (state == self.absorbing_state) or done
        return state, reward, done


    def p(self, next_state, state, action):
        #return self.transition_probas[state, next_state, action]
        return self.transition_probas[next_state, state, action]
    
    
    def r(self, next_state, state, action):
         # if within env boundaries
        if(state < self.n_states-1):
            if self.lake_flat[state] == '$':  # get char of state in environment
                return 1
        return 0


    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '↓', '←', '→']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


    def play(self):
        actions = ['w', 's', 'a', 'd']

        state = self.reset()
        self.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = self.step(actions.index(c))

            self.render()
            print('Reward: {0}.'.format(r))


# ## Small and Big Lakes

# In[7]:


seed = 0

small_lake =    [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]

big_lake =      [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]


# ### Play on small lake

# In[8]:


lake = small_lake
size = len(lake) * len(lake[0])
env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)

#env.play()


# ### Play on Big lake

# In[10]:


#lake = big_lake
#size = len(lake) * len(lake[0])
#env = FrozenLake(lake, slip=0.1, max_steps=size, seed=seed)

#env.play()


# In[ ]:




