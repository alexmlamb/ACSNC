import numpy as np

import sys
from os.path import abspath, join, dirname
sys.path.append(dirname(dirname(abspath(__file__))))

from ctrdapp.optimize.optimizer_factory import create_optimizer
from ctrdapp.scripts.setup_script import setup_script


def full_optimize():
    collision_detector, objects_file, heuristic_factory, initial_guess, configuration, output_path = setup_script()

    optimizer = create_optimizer(heuristic_factory, collision_detector,
                                 np.asarray(initial_guess), configuration)

    optimize_result = optimizer.find_min()

    # possibly rerun solver with a higher number of iterations
    # (especially for rrt* or other optimizing ones) todo

    # interactive?? todo
    optimize_result.save_result(output_path)
    optimize_result.best_solver.visualize_best_solution(objects_file, output_path)
    optimize_result.best_solver.visualize_best_solution_path(objects_file, output_path)


if __name__ == "__main__":
    full_optimize()
