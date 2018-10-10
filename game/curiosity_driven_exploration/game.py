import numpy as np 
import gym
import time

class GameEngine:
    def __init__(self, game_name, ai, channel, verbose=False):
        self.game_name = game_name
        self.ai = ai
        self.verbose = verbose

        self.observations = list()
        self.states = list()
        self.actions = list()
        self.rewards = list()

        self.env = gym.make(self.game_name)
        observation_shape = self.get_observation_shape()
        self.state_shape = *observation_shape, channel

        self.init()

    def get_state_shape(self):
        return np.copy(self.state_shape)

    def get_dataset(self):
        states = self.states[:-1]  # Remove the terminal state
        return states, actions, rewards

    def get_state(self):
        '''
        Get latest state
        '''
        return self.states[-1]

    def update_states(self):
        '''
        Update stored states
        '''
        dimension, channel = self.state_shape
        state = np.zeros((dimension,channel))
        n_observations = len(self.observations)
        for i in range(channel):
            if i+1 <= n_observations:
                state[:,-(i+1)] = self.observations[-(i+1)]

        self.states.append(state)

    def init(self):
        observation = self.env.reset()
        self.observations.append(observation)
        self.update_states()

    def render(self):
        self.env.render()

    def get_observation_shape(self):
        '''
        Get the shape of observation
        '''
        return np.copy(self.env.observation_space.shape)

    def get_action_dimension(self):
        '''
        Get the dimension of action
        '''
        return self.env.action_space.n

    def get_random_action(self):
        return self.env.action_space.sample()

    def step(self, is_random=False):
        if is_random:
            action = self.get_random_action()
        else:
            state = self.get_state()
            action = self.ai.play(state)
        observation, reward, done, info = self.env.step(action)

        self.observations.append(observation)
        self.update_states()
        self.rewards.append(reward)
        self.actions.append(action)

        return done

    def start(self):
        '''
        Visualize the process for AI model to play game
        '''
        while not self.step(is_random=False):
            self.render()
            time.sleep(0.1)

        print("End of game and the final score is [{0}].".format(self.rewards[-1]))

if __name__=="__main__":
    print("Just for test!")
    engine = GameEngine(game_name="Acrobot-v1", ai=None, channel=3, verbose=False)
    print("Action shape: [{0}]".format(engine.get_action_dimension()))
    print("Observation shape: [{0}]".format(engine.get_observation_shape()))
    print("State shape: [{0}]".format(engine.get_state_shape()))

    while not engine.step(is_random=True):
            engine.render()
            time.sleep(0.1)