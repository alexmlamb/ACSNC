from .solver import Solver
from .rrt import RRT
from .rrt_star import RRTStar
from .tree_from_file import TreeFromFile


def create_solver(model, heuristic_factory, collision_detector, configuration) -> Solver:
    """Method for the creation of a (path-planner) Solver object.

    Parameters
    ----------
    model : Model
        Model object containing design info for tube deformation calculation
    heuristic_factory : HeuristicFactory
        Heuristic object containing general parameters and method to create
        new Heuristic object
    collision_detector : CollisionChecker
        Contains obstacles, goal, and methods for collision queries
    configuration : dict
        Dictionary storing configuration variables


    Returns
    -------

    """
    name = configuration.get("solver_type")

    if name == "rrt":
        return RRT(model, heuristic_factory, collision_detector, configuration)
    elif name == "rrt_star":
        return RRTStar(model, heuristic_factory, collision_detector, configuration)
    elif name == "tree_from_file":
        return TreeFromFile(model, heuristic_factory, collision_detector, configuration)
    else:
        raise UserWarning(f"{name} is not a defined solver. "
                          f"Change config file.")
