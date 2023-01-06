'''
Environment where we have a 6x6 grid, and the inner 5x5 region has a block which the agent can push along with it as it moves, but only within 5x5 region.  

agent_pos: 2-vector
block_pos: 2-vector

x,y

render function to image.  1 for edge, 0 for inner part.  

a - left (0), right (1), up (2), down (3), same (4)

'''

#import torch
import numpy as np
import random

class BlockEnv:

    def __init__(self):
        self.agent_pos = [0,0]
        self.block_pos = [1,1]

        self.exo = np.zeros(shape=(36,))
        self.exo_ind = random.randint(0,35)
        self.exo[self.exo_ind] += 1.0

        self.m = 6

    def step(self, a): 

        if a == 0:
            delta = [-1,0]
        elif a == 1:
            delta = [1, 0]
        elif a == 2:
            delta = [0, -1]
        elif a == 3:
            delta = [0, 1]
        elif a == 4:
            delta = [0, 0]

        if self.agent_pos == self.block_pos:
            self.block_pos[0] += delta[0]
            self.block_pos[1] += delta[1]

        self.agent_pos[0] += delta[0]
        self.agent_pos[1] += delta[1]

        if self.block_pos[0] == 0:
            self.block_pos[0] += 1
        if self.block_pos[0] == self.m-1:
            self.block_pos[0] -= 1

        if self.block_pos[1] == 0:
            self.block_pos[1] += 1
        if self.block_pos[1] == self.m-1:
            self.block_pos[1] -= 1

        if self.agent_pos[0] == -1:
            self.agent_pos[0] += 1
        if self.agent_pos[0] == self.m:
            self.agent_pos[0] -= 1

        if self.agent_pos[1] == -1:
            self.agent_pos[1] += 1
        if self.agent_pos[1] == self.m:
            self.agent_pos[1] -= 1

        self.exo = np.zeros(shape=(36,))
        self.exo_ind = random.randint(0,35)
        self.exo[self.exo_ind] += 1.0

    def get_obs(self):
        x = np.zeros(shape=(self.m, self.m))

        for j in range(self.m):
            for i in range(self.m):
                if self.agent_pos == [i,j] and self.block_pos == [i,j]:
                    x[j,i] += 3.0
                elif self.agent_pos == [i,j]:
                    x[j,i] += 1.0
                elif self.block_pos == [i,j]:
                    x[j,i] += 2.0

        agent_state = self.agent_pos[0]*self.m + self.agent_pos[1]
        block_state = self.block_pos[0]*self.m + self.block_pos[1]

        x = np.concatenate([x, self.exo.reshape((6,6))], axis=1)

        return x.flatten(), agent_state, block_state, self.exo_ind

    def render(self):

        for j in range(self.m):
            for i in range(self.m):
                if self.agent_pos == [i,j] and self.block_pos == [i,j]:
                    c = "X"
                elif self.agent_pos == [i,j]:
                    c = 'x'
                elif self.block_pos == [i,j]:
                    c = 'b'
                else:
                    c = '.'
                print(c,end=' ')
            print("")


if __name__ == "__main__":

    env = BlockEnv()

    import random
    for i in range(0,5000):
        a = random.randint(0,4)
        env.render()
        env.step(a)
        print(a)



