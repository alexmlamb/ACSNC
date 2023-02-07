import gym
from gym import spaces
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
import cv2
import numpy as np


class RatGym(gym.Env):
    """
    Gym wrapper for RatInABox
    """

    def __init__(self):
        self.env = Environment()
        self.agent = Agent(self.env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(100, 100, 3), dtype=np.float32
        )

    def render(self):
        grid_size = 200
        base = np.zeros((grid_size, grid_size, 3), dtype=np.uint8) + 200
        for wall in self.env.walls:
            wall = wall * grid_size
            cv2.line(
                base,
                (int(wall[0][0]), int(wall[0][1])),
                (int(wall[1][0]), int(wall[1][1])),
                (100, 100, 100),
                5,
            )
        agent_pos = self.agent.pos * grid_size
        cv2.circle(base, (int(agent_pos[0]), int(agent_pos[1])), 3, (50, 50, 200), -1)
        base = cv2.resize(base, (100, 100)) / 255.0
        return base

    def reset(self, start_pos=None):
        self.agent = Agent(self.env)
        if start_pos is not None:
            self.agent.pos = start_pos
        else:
            self.agent.pos = np.array([0.5, 0.5])
        return self.render().reshape(-1)

    def step(self, action):
        action = action * 100
        self.agent.update(drift_velocity=action)
        return (
            self.render().reshape(-1),
            0,
            False,
            {"pos": self.agent.pos, "vel": self.agent.velocity},
        )

    def add_wall(self, wall):
        self.env.add_wall(wall)
