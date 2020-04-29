"""
Taken from
https://github.com/xupe/mistake-in-retro-contest-of-OpenAI/blob/master/src/utils/sonic_util.py
and modified to suit our motivations.
"""

import gym
import optparse
import sys
import os
import random
import numpy as np




class Discretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], ['UP']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class AllowBacktracking(gym.Wrapper):
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.rings = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        self.rings = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        #rew = self._cur_x - self._max_x
        self._max_x = max(self._max_x, self._cur_x)

        new_rings = info['rings']
        if(new_rings == 0 and self.rings != 0):
            rew -= 2000
        elif(new_rings < self.rings):
            rew -= 1000
        else:
            rew += (new_rings - self.rings)*200
        self.rings = new_rings

        return obs, rew, done, info


class Discretizer2(gym.ActionWrapper):
    
    def __init__(self, env):
        super(Discretizer2, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], ['UP'], ['UP', 'RIGHT'], ['RIGHT', 'B']] 
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class AllowBacktracking2(gym.Wrapper):
    def __init__(self, env):
        super(AllowBacktracking2, self).__init__(env)
        self.cur_x = 0
        self.cur_y = 0
        self.max_x = 0
        self.rings = 0
        self.score = 0
        self.past_x = [0]*10
        self.counter = 0
        self.past_screen_y = [1440]*25

    def reset(self, **kwargs): # pylint: disable=E0202
        self.cur_x = 0
        self.cur_y = 0
        self.max_x = 0
        self.rings = 0
        self.score = 0
        self.past_x = list(range(-19,1,1))
        self.counter = 0
        self.past_screen_y = [1440]*25
        
        # print(self.past_x)
        # input()
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        new_score = info['score']
        new_rings = info['rings']
        
        reward = rew*2
        # print(action)
        
        if done:
            reward += -2000
        # print(rew,info)
        self.counter+=1
        # print(self.counter)
        # input()
        if new_score != self.score:
            # input('score change')
            reward += 5*(new_score - self.score)
            self.score = new_score
            
        if new_rings != self.rings:
            reward += 100*(new_rings - self.rings)
            self.rings = new_rings
            # input('rings change')

        self.cur_x = info['x']
        # self.cur_y = info['y']
        self.past_x = self.past_x[1:]
        self.past_x.append(self.cur_x)
        
        dist = self.cur_x - self.past_x[0]
        reward += dist/20

        dist_add = 2*(self.cur_x - self.max_x)/3
        if dist_add < 0:
            dist_add/=4
        reward += dist_add
        self.max_x = max(self.cur_x, self.max_x)
        
        mean_screen_y = sum(self.past_screen_y)/len(self.past_screen_y)
        cur_screen_y = info['screen_y']
        # reward += abs(mean_screen_y - cur_screen_y)/6
        
        self.past_screen_y.append(cur_screen_y)
        self.past_screen_y = self.past_screen_y[1:]
        
        # print('rew',rew*2,'score',70*(new_score - self.score),'rings',100*(new_rings - self.rings))
        # print('dist',dist_add, 'y',abs(mean_screen_y - cur_screen_y)/6, 'dist',dist/20)
        
        
        if action in [1,3,8,9] and reward < 5:
            reward-=20
        
        return obs, reward, done, info