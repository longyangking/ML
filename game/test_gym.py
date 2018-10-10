import numpy as np 
import os 

# from keras.models import Model 
# from keras.layers import Input, Dense

import gym

# def build_model(input_shape, action_dim):
#     input_tensor = 

env = gym.make('SpaceInvaders-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        print(dir(env.action_space))
        #print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break