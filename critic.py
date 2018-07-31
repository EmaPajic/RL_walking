"""
@author: user
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, InputLayer, Concatenate
from keras.optimizers import Adam

def create_critic(observation, action):
    flattened_observation = Flatten()(observation)
    output = Dense(400)(flattened_observation)
    output = Activation('relu')(output)
    output = Concatenate()([output, action])
    output = Dense(300)(output)
    output = Activation('relu')(output)
    output = Dense(1)(output)
    output = Activation('linear')(output)
    critic = Model([action, observation], outputs=output)
    print(critic.summary())
    return critic