"""Package for creating types of heuristics."""

from abc import ABC, abstractmethod
from .obstacles_and_goal import SquareObstacleAvgPlusWeightedGoal
from .only_goal_distance import OnlyGoalDistance
from .follow_the_leader import FollowTheLeader, FollowTheLeaderWInsertion, FollowTheLeaderTranslation
from .heuristic import Heuristic


class HeuristicFactory(ABC):
    """Class for the creation of a heuristic."""

    @abstractmethod
    def create(self, **kwargs) -> Heuristic:
        """Creates a Heuristic using the passed named arguments.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments with the heuristic configuration parameters

        Returns
        -------
        Heuristic
        """
        pass

    @abstractmethod
    def create_from_old(self, heuristic: Heuristic, **kwargs) -> Heuristic:
        """Creates a Heuristic based off of given heuristic and named arguments.

        Parameters
        ----------
        heuristic : Heuristic
            The old heuristic to use for certain parameters
        **kwargs
            Arbitrary keyword arguments with the heuristic configuration parameters

        Returns
        -------
        Heuristic
        """
        pass


class SOAPWGFactory(HeuristicFactory):
    """Factory to create a SOAPWG heuristic.

    Parameters
    ----------
    goal_weight : float
        The weight of the goal with respect to the obstacles cost
    *args
        Variable length argument list

    Attributes
    ----------
    goal_weight : float
        The weight of the goal with respect to the obstacles cost
    """

    def __init__(self, goal_weight, *args):
        self.goal_weight = goal_weight

    def create(self, **kwargs) -> SquareObstacleAvgPlusWeightedGoal:
        # uses inputs of min_obstacle_distance and goal_distance
        return SquareObstacleAvgPlusWeightedGoal(self.goal_weight,
                                                 kwargs.get('min_obstacle_distance'),
                                                 kwargs.get('goal_distance'))

    def create_from_old(self, heuristic: SquareObstacleAvgPlusWeightedGoal, **kwargs):
        return self.create(min_obstacle_distance=heuristic.store_obstacle_min,
                           goal_distance=heuristic.goal_dist)


class OnlyGoalDistanceFactory(HeuristicFactory):
    """Factory to create a Only Goal Distance heuristic.

    Parameters
    ----------
    *args
        Variable length argument list
    """

    def __init__(self, *args):
        pass

    def create(self, **kwargs) -> OnlyGoalDistance:
        # only uses goal_distance input
        return OnlyGoalDistance(kwargs.get('goal_distance'))

    def create_from_old(self, heuristic: OnlyGoalDistance, **kwargs) -> OnlyGoalDistance:
        # goal distance is stored as cost and never mutated
        return self.create(goal_distance=heuristic.get_cost())


class FTLFactory(HeuristicFactory):
    """Factory to create a Follow The Leader heuristic.

    Parameters
    ----------
    only_tip : bool
        Should only the tip be used for calculated the FTL cost?
    *args
        Variable length argument list

    Attributes
    ----------
    only_tip : bool
        Should only the tip be used for calculated the FTL cost?
    """

    def __init__(self, only_tip, *args):
        self.only_tip = only_tip

    def create(self, **kwargs) -> FollowTheLeader:
        # only uses follow_the_leader input
        return FollowTheLeader(self.only_tip, kwargs.get('follow_the_leader'))

    def create_from_old(self, heuristic: FollowTheLeader, **kwargs) -> FollowTheLeader:
        # follow the leader will be updated in the arguments
        return self.create(**kwargs)


class FTLWInsertionFactory(FTLFactory):
    """Factory to create a Follow The Leader with Insertion heuristic.

    Parameters
    ----------
    only_tip : bool
        Should only the tip be used for calculated the FTL cost?
    insertion_weight : float
        The weight of the insertion cost with respect to the FTL cost
    *args
        Variable length argument list

    Attributes
    ----------
    only_tip : bool
        Should only the tip be used for calculated the FTL cost?
    insertion_weight : float
        The weight of the insertion cost with respect to the FTL cost
    """

    def __init__(self, only_tip, insertion_weight, *args):
        super().__init__(only_tip, *args)
        self.insertion_weight = insertion_weight

    def create(self, **kwargs) -> FollowTheLeaderWInsertion:
        # only uses follow_the_leader input
        return FollowTheLeaderWInsertion(self.only_tip, self.insertion_weight,
                                         kwargs.get('follow_the_leader'),
                                         kwargs.get('insertion_fraction'))

    def create_from_old(self, heuristic: FollowTheLeaderWInsertion, **kwargs) -> FollowTheLeaderWInsertion:
        return self.create(follow_the_leader=kwargs.get('follow_the_leader'),
                           insertion_fraction=heuristic.insertion_fraction)


class FTLTranslationFactory(FTLFactory):

    def __init__(self, only_tip, *args):
        super().__init__(only_tip, *args)

    def create(self, **kwargs) -> FollowTheLeaderTranslation:
        return FollowTheLeaderTranslation(self.only_tip, kwargs.get('follow_the_leader'))

    def create_from_old(self, heuristic: FollowTheLeaderTranslation, **kwargs) -> FollowTheLeaderTranslation:
        # follow the leader will be updated in the arguments
        return self.create(**kwargs)


def create_heuristic_factory(configuration, heuristic_dict) -> HeuristicFactory:
    """Creates the heuristic factory based on the given configuration dictionary.

    Parameters
    ----------
    configuration : dict
        The configuration for this run, including heuristic_type
    heuristic_dict : dict
        Dictionary of the required parameters of the heuristics

    Returns
    -------
    HeuristicFactory
    """
    name = configuration.get("heuristic_type")
    param_names = heuristic_dict.get(name)
    params = [configuration.get(n) for n in param_names]

    if name == "square_obstacle_avg_plus_weighted_goal":
        return SOAPWGFactory(*params)
    elif name == "only_goal_distance":
        return OnlyGoalDistanceFactory(*params)
    elif name == "follow_the_leader":
        return FTLFactory(*params)
    elif name == "follow_the_leader_w_insertion":
        return FTLWInsertionFactory(*params)
    elif name == "follow_the_leader_translation":
        return FTLTranslationFactory(*params)
    else:
        raise UserWarning(f"{name} is not a defined heuristic. "
                          f"Change config file.")
