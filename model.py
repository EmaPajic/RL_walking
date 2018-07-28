"""
@author: Ema & Sofia
"""

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input
from keras import initializers

class Actor:
    
    def __init__(self, state_size, action_size, seed, fc_units=256):
        model = Sequential()
        lim = 1. / np.sqrt(state_size)
        model.add(Dense(fc_units, activation='relu', input_shape=(state_size,), kernel_initializer=initializers.random_uniform(minval=-lim,maxval=lim,seed=seed)))
        model.add(Dense(action_size, activation='tanh', input_shape=(fc_units,), kernel_initializer=initializers.random_uniform(minval=-3e-3,maxval=3e-3,seed=seed)))
        self.model = model
        
    def loss_function():
        return 0
    
class Critic:
    
    def __init__(self, observation, action, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        model = Sequential()
        lim1 = 1. / np.sqrt(state_size)
        lim2 = 1. / np.sqrt(fcs1_units+action_size)
        lim3 = 1. / np.sqrt(fc2_units)
        model.add(Dense(fcs1_units, activation='relu', input_shape=(state_size,),kernel_initializer=initializers.random_uniform(minval=-lim1,maxval=lim1,seed=seed)))
        
        ac = Input((action_size,))
        #print(model.type)
        out = Concatenate([model, ac])
        model.add(Dense(fc2_units, activation='relu', input_shape=(fcs1_units+action_size,), kernel_initializer=initializers.random_uniform(minval=-lim2,maxval=lim2,seed=seed)))
        model.add(Dense(fc3_units, activation='relu', input_shape=(fc2_units,),kernel_initializer=initializers.random_uniform(minval=-lim3,maxval=lim3,seed=seed)))        
        model.add(Dense(1, activation='tanh', input_shape=(fc3_units,),kernel_initializer=initializers.random_uniform(minval=-3e-3,maxval=3e-3,seed=seed)))
        self.model = Model(inputs=[action, observation], outputs=model)
        
    def loss_function():
        return 0

actor = Actor(5,5,2)
critic = Critic(2,2,5,5,2)