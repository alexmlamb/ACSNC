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
        for state in self.state:
            for next_state in self.next_state:
                if not np.isnan(
                        self.transition[self.unique_states_dict[state], self.unique_states_dict[next_state], 0]):
                    edges.append((state, next_state))

        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        if save_path is not None:
            plt.savefig(save_path)
        return graph
