#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:00:08 2018

@author: user
"""

import numpy as np

import gym
from gym import wrappers

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from bipedal_processor import BipedalProcessor
from actor import create_actor
from critic import create_critic
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = 'BipedalWalker-v2'
gym.undo_logger_setup()

def main():
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, '/tmp/{}'.format(ENV_NAME), force=True)
    np.random.seed(123)
    env.seed(123)
    assert len(env.action_space.shape) == 1
    
    action_shape = env.action_space.shape[0]
    observation_shape = env.observation_space.shape

    actor = create_actor(observation_shape, action_shape)
    
    action_input = Input(shape=(action_shape,), name='action_input')
    observation_input = Input(shape=(1,) + observation_shape, name='observation_input')
    
    critic = create_critic(observation_input, action_input)

    memory = SequentialMemory(limit=100000, window_length=1)
    
    random_process = OrnsteinUhlenbeckProcess(size=action_shape, theta=.15, mu=0., sigma=.1)
    agent = DDPGAgent(nb_actions=action_shape, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=BipedalProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])
    agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))
    #agent.fit()
    agent.fit(env, nb_steps=200, action_repetition = 100, visualize=True, verbose=1)
    agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
    #agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
    
main()