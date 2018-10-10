import numpy as np 
from ai import AI 
from game import GameEngine
import time

class TrainAI:
    def __init__(self, game_name, channel, ai=None, verbose=False):
        self.game_name = game_name
        self.channel = channel
        self.verbose = verbose
        
        if ai is not None:
            self.ai = ai
        else:
            self.ai = AI(
                game_name=game_name, 
                channel=channel, 
                verbose=verbose
            )

    def get_selfplay_data(self, n_rounds, max_step=100, epsilon=0.5):
        states = list()
        actions = list()
        rewards = list()

        if self.verbose:
            start_time = time.time()

        for n in range(n_rounds):
            if self.verbose:
                print("{0}th round of self-play process...".format(n+1))

            engine = GameEngine(
                game_name=self.game_name, 
                ai=self.ai, 
                channel=self.channel, 
                verbose=self.verbose)

            step = 0 
            flag = False
            while (not flag) and (step < max_step):
                v = np.random.random()
                is_random = False
                if v < epsilon:
                    is_random = True
                flag = engine.step(is_random)
                step += 1

            _states, _actions, _rewards = engine.get_dataset()

            for i in range(len(_rewards)):
                states.append(_states[i])
                actions.append(_actions[i])
                rewards.append(_rewards[i])

        if self.verbose:
            endtime = time.time()
            print("End of self-play process with data size [{0}] and cost time [{1:.1f}s].".format(len(states),(endtime - starttime)))

        return states, actions, rewards

    def update_ai(self, dataset):
        pass

    def start(self, filename):
        pass
