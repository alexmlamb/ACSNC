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
        self.agent_pos = [0, 0, 1]
        

        self.m = 4

        self.block_pos = [self.m - 1, 0, 1]

        #self.exo = np.zeros(shape=(self.m**2,))
        #self.exo_ind = random.randint(0,self.m**2 - 1)
        #self.exo[self.exo_ind] += 1.0

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

        if self.block_pos[1] != self.agent_pos[1]:
            self.agent_pos[0] += delta[0]
            self.agent_pos[1] += delta[1]

        #if self.agent_pos == self.block_pos:
        #    self.agent_pos[2] = (self.agent_pos[2] + 1) % 4 + 1
        """
        if self.block_pos[0] % 2 == 1:
           self.block_pos[1] -= 1
        elif self.block_pos[0] % 2 == 0:
           self.block_pos[1] += 1
            

        if self.block_pos[1] == -1:
            if self.block_pos[0] != 0:
                self.block_pos[0] -= 1#self.m - 1
                self.block_pos[1] = 0
            else:
                self.block_pos[0] += 1
                self.block_pos[1] = 0


        if self.block_pos[1] == self.m:
            if self.block_pos[0] != 0:
                self.block_pos[0] -= 1
                self.block_pos[1] = self.m - 1
            else:
                self.block_pos[0] += 1
                self.block_pos[1] = self.m - 1
        """
        """
        if self.block_pos[1] == -1:
            if self.block_pos[0] != 0:
                self.block_pos[0] -= 1#self.m - 1
                self.block_pos[1] = 0
            else:
                self.block_pos[0] += 1
                self.block_pos[1] = 0


        if self.block_pos[1] == self.m:
            if self.block_pos[0] != 0:
                self.block_pos[0] -= 1
                self.block_pos[1] = self.m - 1
            else:
                self.block_pos[0] += 1
                self.block_pos[1] = self.m - 1
        """

        block_move_direction = random.randint(0, 1)
        if block_move_direction == 0:
            if self.block_pos[1] != 0:
                self.block_pos[1] -= 1
        elif block_move_direction == 1:
            if self.block_pos[1] != self.m - 1:
                self.block_pos[1] += 1


        if self.agent_pos[0] == -1:
            self.agent_pos[0] = self.m - 2
        if self.agent_pos[0] == self.m - 1:
            self.agent_pos[0] = 0

        if self.agent_pos[1] == -1:
            self.agent_pos[1] = self.m - 1
        if self.agent_pos[1] == self.m:
            self.agent_pos[1] = 0

        #if self.agent_pos == self.block_pos:
        #    self.agent_pos[2] = (self.agent_pos[2] + 1) % 4 + 1

        

        

    def get_obs(self):
        x = np.zeros(shape=(self.m, self.m))
        
        x[self.agent_pos[0], self.agent_pos[1]] = 1
        x[self.block_pos[0], self.block_pos[1]] = 2 #self.block_pos[-1]
        
        return x[:, :, None], self.agent_pos[0] * self.m + self.agent_pos[1], self.block_pos[0] * self.m + self.block_pos[1]

    def render(self):

        x, _, _ = self.get_obs()

        print(x[:, :])

if __name__ == "__main__":

    env = BlockEnv()

    import random
    for i in range(0,5000):
        a = random.randint(0,4)
        env.render()
        
        env.step(a)



