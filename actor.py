"""
@author: user
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

def create_actor(observation_shape, action_shape):
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + observation_shape))
    actor.add(Dense(400))
    actor.add(Activation('relu'))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(action_shape))
    actor.add(Activation('tanh'))
    print(actor.summary())
    return actor