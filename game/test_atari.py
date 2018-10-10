from __future__ import print_function
import sys, gym, time

env = gym.make('SpaceInvaders-v0')

human_agent_action = 0
ACTIONS = env.action_space.n

def key_press(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release

def rollout(env):
    global human_agent_action
    obser = env.reset()

    while 1:
        obser, r, done, info = env.step(human_agent_action)
        if r != 0:
            print("reward %0.3f" % r)
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        time.sleep(0.1)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

rollout(env)