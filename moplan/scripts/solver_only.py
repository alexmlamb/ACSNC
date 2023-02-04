from ctrdapp.scripts.setup_script import setup_script
from ctrdapp.model.model_factory import create_model
from ctrdapp.solve.solver_factory import create_solver

import time


def solver_only():
    collision_detector, objects_file, heuristic_factory, desired_q, configuration, output_path = setup_script()

    this_model = create_model(configuration, desired_q)

    iter_num = input("How many iterations? ")
    for n in range(int(iter_num)):

        this_output_path = output_path / f"Run{n}"
        this_output_path.mkdir(parents=True)

        start_time = time.time()

        this_solver = create_solver(this_model, heuristic_factory, collision_detector, configuration)

        full_time = time.time() - start_time
        print(f"Run took {full_time} sec(?)")

        # call get_best_cost
        cost, best_ind = this_solver.get_best_cost()
        print(cost)

        this_solver.visualize_full_search(this_output_path, with_solution=True)
        this_solver.save_tree(this_output_path)
        this_solver.save_best_solution(this_output_path)
        this_solver.visualize_best_solution(objects_file, output_path)
        this_solver.visualize_best_solution_path(objects_file, output_path)


if __name__ == "__main__":
    solver_only()
