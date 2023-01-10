'''
    Environment where agent moves in the (0,255), where the state is kept to always be divisible by 256.  

    agent_pos: 2-vector

    x,y

    render function to image.  Place a pixel in a 256x256 image where the agent is.  

    a - left (0), right (1), up (2), down (3), same (4)

'''

import numpy as np
import random

def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = int(e*m)
        e = e/m
        x[i] = e
    return x

class RoomEnv:

    def __init__(self):
        self.agent_pos = [0,0]

    def random_action(self): 

        delta = [0, 0]
        delta[0] = random.uniform(-0.1, 0.1)
        delta[1] = random.uniform(-0.1, 0.1)

        return div_cast(delta)

    def step(self, a):

        self.agent_pos = div_cast(self.agent_pos)

        self.agent_pos[0] += a[0]
        self.agent_pos[1] += a[1]

        if self.agent_pos[0] <= 0:
            self.agent_pos[0] = 0
        if self.agent_pos[0] >= 1:
            self.agent_pos[0] = 1

        if self.agent_pos[1] <= 0:
            self.agent_pos[1] = 0
        if self.agent_pos[1] >= 1:
            self.agent_pos[1] = 1

        self.agent_pos = div_cast(self.agent_pos)

    def get_obs(self):
        x = np.zeros(shape=(100, 100))

        x[self.agent_pos[0]*100, self.agent_pos[1]*100] += 1

        #x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        return x.flatten(), self.agent_pos


if __name__ == "__main__":

    env = RoomEnv()

    for i in range(0,5000):
        a = env.random_action()
        print('s', env.agent_pos)
        print('a', a)
        env.step(a)



