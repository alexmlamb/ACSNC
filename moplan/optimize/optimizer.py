"""The optimizer class."""

from abc import ABC, abstractmethod

from .optimize_result import OptimizeResult


class Optimizer(ABC):
    """Runs an optimization function on the design parameters: q.

    Parameters
    ----------
    heuristic_factory : HeuristicFactory
        Heuristic object containing general parameters and method to create
        new Heuristic object
    collision_checker : CollisionChecker
        Contains obstacles, goal, and methods for collision queries
    initial_guess : np.ndarray
        Initial guess to start the optimizer with
    configuration : dict
        Dictionary storing configuration variables

    Attributes
    ----------
    tube_num : int
        Number of tubes
    precision : float
        Tolerance for algorithm termination
    configuration : dict
        Dictionary storing configuration variables
    heuristic_factory : HeuristicFactory
        Heuristic object containing general parameters and method to create
        new Heuristic object
    collision_checker : CollisionChecker
        Contains obstacles, goal, and methods for collision queries
    initial_guess : np.ndarray
        Initial guess to start the optimizer with
    """

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        self.tube_num = configuration.get("tube_number")
        self.precision = configuration.get("optimizer_precision")
        self.configuration = configuration
        self.heuristic_factory = heuristic_factory
        self.collision_checker = collision_checker
        self.initial_guess = initial_guess

    @abstractmethod
    def find_min(self):
        """Return the result from running the optimization algorithm.

        Returns
        -------
        OptimizeResult
            The result of the optimize procedure, containing solution, run
            parameters and best solver.
        """
        pass


