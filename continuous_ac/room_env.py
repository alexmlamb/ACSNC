'''
    Environment where agent moves in the (0,255), where the state is kept to always be divisible by 256.  

    agent_pos: 2-vector

    x,y

    render function to image.  Place a pixel in a 256x256 image where the agent is.  

    a - left (0), right (1), up (2), down (3), same (4)

'''
import gym
import numpy as np
import random
from gym import spaces


def div_cast(x, m=100):
    for i in range(len(x)):
        e = x[i]
        e = round(e * m)
        e = e / m
        x[i] = e
    return x


class RoomEnv(gym.Env):

    def __init__(self):
        self.agent_pos = [0, 0]
        self.action_space = spaces.Box(-0.1, 0.1, (2,), dtype=np.float32)

    def reset(self):
        self.agent_pos = [0, 0]

        obs, agent_pos, exo = self._get_obs()
        info = {'agent_pos': agent_pos, 'exo': exo}
        return obs, info

    def random_action(self):

        delta = [0, 0]
        delta[0] = random.uniform(-0.1, 0.1)
        delta[1] = random.uniform(-0.1, 0.1)

        return div_cast(delta)

    def step(self, a):

        self.agent_pos = self.agent_pos[:]

        self.agent_pos = div_cast(self.agent_pos)

        self.agent_pos[0] += a[0]
        self.agent_pos[1] += a[1]

        if self.agent_pos[0] <= 0:
            self.agent_pos[0] = 0
        if self.agent_pos[0] >= 1:
            self.agent_pos[0] = 0.99

        if self.agent_pos[1] <= 0:
            self.agent_pos[1] = 0
        if self.agent_pos[1] >= 1:
            self.agent_pos[1] = 0.99

        self.agent_pos = div_cast(self.agent_pos)[:]

        done = False
        reward = 0

        next_obs, agent_pos, exo = self._get_obs()
        info = {'agent_pos': agent_pos, 'exo': exo}

        return next_obs, reward, done, info

    def synth_obs(self, ap):
        x = np.zeros(shape=(100, 100))

        x[int(round(ap[0] * 100)), int(round(ap[1] * 100))] += 1

        return x.flatten()

    def _get_obs(self):
        x = np.zeros(shape=(100, 100))
        x[int(round(self.agent_pos[0] * 100)), int(round(self.agent_pos[1] * 100))] += 1
        # x = np.concatenate([x, self.exo.reshape((self.m,self.m))], axis=1)

        exo = [0.0, 0.0]

        return x.flatten(), self.agent_pos, exo

    def get_obs(self):
        return self._get_obs()


if __name__ == "__main__":

    env = RoomEnv()
    obs, info = env.reset()
    for i in range(0, 5000):
        action = env.random_action()
        next_obs, reward, done, info = env.step(action)

        print('agent-pos:', info['agent_pos'])
        print('exo:', info['exo'])
        print('obs-argmax: ', next_obs.argmax())
