"""The Heuristic class using only the goal distance."""

from .heuristic import Heuristic


class OnlyGoalDistance(Heuristic):
    """Heuristic using only the distance from the goal.

    Parameters
    ----------
    goal_dist : float
        The distance from the goal for this Heuristic

    Attributes
    ----------
    cost : float
        The cost for this Heuristic
    """

    def __init__(self, goal_dist):
        self.cost = goal_dist

    def get_cost(self):
        return self.cost

    def calculate_cost_from_parent(self, parent: "OnlyGoalDistance", reset=False, init_insertion=False):
        pass

    def test_cost_from_parent(self, parent: "OnlyGoalDistance"):
        print("Parent doesn't effect cost of Only Goal Distance heuristic.")
        return self.cost
