"""Factory method to create an optimizer."""

import numpy as np

from .optimizer import Optimizer
from .neldermead import NelderMead
from ctrdapp.heuristic.heuristic_factory import HeuristicFactory
from ctrdapp.collision.collision_checker import CollisionChecker


def create_optimizer(heuristic_factory, collision_checker, initial_guess, configuration) -> Optimizer:
    """Creates an optimizer based on the given configuration and with given objects.

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

    Returns
    -------
    Optimizer
        The desired optimizer object
    """
    name = configuration.get("optimizer_type")

    if name == "nelder_mead":
        return NelderMead(heuristic_factory, collision_checker, initial_guess, configuration)
    else:
        raise UserWarning(f"{name} is not a defined optimizer. "
                          f"Change config file.")
