import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class EmpiricalMDP:

    def __init__(self, state, action, next_state, reward):
        self.unique_states = sorted(np.unique(np.concatenate((state, next_state), axis=0)))
        self.unique_states_dict = {k: i for i, k in enumerate(self.unique_states)}
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.transition = self.__estimate_transition()

        self.discrete_transition = self.__discrete_transition()

    def __discrete_transition(self):

        # discretize actions
        actions = []
        for x in np.arange(-0.1, 0.1 + 0.01, 0.01):
            for y in np.arange(-0.1, 0.1 + 0.01, 0.01):
                actions.append((round(x, 2), round(y, 2)))
        self.discrete_action_space = np.unique(actions, axis=0)

        # generate discrete transition matrix containing visit count
        action_value_idx_map = {tuple(val): idx for idx, val in enumerate(self.discrete_action_space)}
        transition = np.zeros((len(self.unique_states), len(self.discrete_action_space), len(self.unique_states)))
        for state in range(len(self.transition)):
            for next_state, action in enumerate(self.transition[state]):
                if not np.isnan(action).all():
                    transition[state][action_value_idx_map[tuple(np.round(action, 2))]][next_state] += 1

        return transition

    def __estimate_transition(self):
        transition = np.empty((len(self.unique_states), len(self.unique_states), len(self.action[0])))
        transition[:, :, :] = np.nan
        for state in self.unique_states:
            for next_state in self.unique_states:
                _filter = np.logical_and(self.state == state,
                                         self.next_state == next_state)
                if True in _filter:
                    transition[self.unique_states_dict[state], self.unique_states_dict[next_state], :] = self.action[
                        _filter].mean(axis=0)
        return transition

    def visualize_transition(self, save_path=None):
        graph = nx.DiGraph()
        edges = []
        for state in self.unique_states:
            for next_state in self.unique_states:
                if not np.isnan(
                        self.transition[self.unique_states_dict[state], self.unique_states_dict[next_state], 0]):
                    edges.append((state, next_state))

        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        return graph

    def visualize_path(self, path, save_path=None):
        graph = nx.DiGraph()
        edges = []
        for state in self.unique_states:
            for next_state in self.unique_states:
                if not np.isnan(
                        self.transition[self.unique_states_dict[state], self.unique_states_dict[next_state], 0]):
                    edges.append((state, next_state))

        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        return graph

    def step(self, state, action_idx):
        """ samples a next state from current state and action"""
        next_state_visit_count = self.discrete_transition[state][action_idx]
        next_state_prob = self.__normalize(next_state_visit_count, next_state_visit_count.min(),
                                           next_state_visit_count.max())
        next_state_sample = np.random.choice(np.arange(0, len(next_state_visit_count)), 1, replace=False,
                                             p=next_state_prob)

        return next_state_sample[0]

    @staticmethod
    def __normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr
