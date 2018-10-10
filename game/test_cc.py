import numpy as np 
import os 

# from keras.models import Model 
# from keras.layers import Input, Dense

import gym
env = gym.make('Acrobot-v1')

print(env.observation_space.shape)