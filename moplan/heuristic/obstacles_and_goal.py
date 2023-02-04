"""Package for heuristic with obstacles and goal distance."""

from .heuristic import Heuristic


class SquareObstacleAvgPlusWeightedGoal(Heuristic):
    """The heuristic for the squared inverse obstacle min with goal distance.

    This cost is given by the average of the squared inverse obstacle minimums
    plus the distance from the goal (weighted with respect to the obstacle average).
    The cost decreases as the curve tip gets closer to the goal and the whole
    curve stays away from obstacles over the entire path.

    Parameters
    ----------
    goal_weight : float
        Weight of the goal cost with respect to the obstacle cost
    this_obstacle_min : float
        The minimum distance of this curve to the nearest obstacle
    goal_dist : float
        The distance from the tip of the curve to the goal

    Attributes
    ----------
    store_obstacle_min : float
        Stores the actual minimum distance to the nearest obstacle
    this_obstacle_min : float
        Stores the squared inverse of the obstacle minimum
    avg_obstacle_min : float
        Stores the average modified obstacle minimum of the path
    generation : int
        The generation of this Heuristic (+1 from its parent's)
    goal_weight : float
        Weight of the goal cost with respect to the obstacle cost
    goal_dist : float
        The distance from the tip of the curve to the goal
    """

    def __init__(self, goal_weight, this_obstacle_min, goal_dist):
        self.store_obstacle_min = this_obstacle_min
        self.this_obstacle_min = (1 / this_obstacle_min)**2
        self.avg_obstacle_min = self.this_obstacle_min
        self.generation = 0
        self.goal_weight = goal_weight
        self.goal_dist = goal_dist

    def calculate_cost_from_parent(self, parent: 'SquareObstacleAvgPlusWeightedGoal', reset=False, init_insertion=False):
        if self.generation != 0 and not reset:
            print(f"Cost already calculated. Do not run method twice.")
            return
        self.avg_obstacle_min = self._calculate_avg(parent.avg_obstacle_min,
                                                    parent.generation,
                                                    self.this_obstacle_min)
        self.generation = parent.generation + 1

        if init_insertion:
            self.generation = 0

    def test_cost_from_parent(self, parent: "SquareObstacleAvgPlusWeightedGoal"):
        store_avg = self.avg_obstacle_min
        self.avg_obstacle_min = self._calculate_avg(parent.avg_obstacle_min, parent.generation, self.this_obstacle_min)
        cost = self.get_cost()
        self.avg_obstacle_min = store_avg
        return cost

    @staticmethod
    def _calculate_avg(parent_avg, parent_gen, this_obstacle_min):
        """Calculate the average based on the given parameters.

        Parameters
        ----------
        parent_avg : float
            average obstacle minimum value of the parent heuristic
        parent_gen : int
            generation of the parent heuristic
        this_obstacle_min : float
            obstacle minimum value of this heuristic

        Returns
        -------
        float
            new average value using parent average and this obstacle min
        """
        prior_sum = parent_gen * parent_avg
        return (prior_sum + this_obstacle_min) / (parent_gen + 1)

    def get_cost(self):
        return self.goal_dist * self.goal_weight + self.avg_obstacle_min
