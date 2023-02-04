"""Container for the result of an Optimizer."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pathlib

from ctrdapp.solve.solver import Solver


class OptimizeResult:
    """Packages pertinent results of the optimizer find_min function.

    Parameters
    ----------
    best_q : np.ndarray
        the optimal coordinates
    best_solver : Solver
        the optimal solver (to give access to visualization tools)
    success : bool
        did the optimization converge?
    num_function_eval : int
        number of function (solver) evaluations
    num_iterations : int
        number of iterations of optimizer
    completion_time : float
        total time for optimizer to run (in sec)
    optimize_process : list[dict]
        list of dicts{q, cost} recording the process of the optimizer
    """

    def __init__(self, best_q, best_solver, success, num_function_eval,
                 num_iterations, completion_time, optimize_process):
        self.best_q = best_q
        self.best_solver = best_solver
        self.success = success
        self.num_function_eval = num_function_eval
        self.num_iterations = num_iterations
        self.completion_time = completion_time
        self.optimize_process = optimize_process  # list of dicts{q, cost}

    def graph_process(self):  # todo
        fig = plt.figure()
        line, = plt.plot([], [], 'r-')
        optimize_iterator = [node_dict.get('q') for node_dict in self.optimize_process]
        optimize_iterator = iter(optimize_iterator)

        def update(data):
            line.set_xdata(data[0])
            line.set_ydata(data[1])
            return line,

        def data_gen():
            while True:
                yield next(optimize_iterator)

        plt.xlim(0, 0.1)
        plt.ylim(0, 0.1)
        plt.title('test')
        line_ani = animation.FuncAnimation(fig, update, data_gen, blit=True, interval=50)
        # writer = FFMpegWriter(fps=15, metadata=dict(artist='Conor'), bitrate=1800)
        # line_ani.save("mygraph.mp4", writer=writer)

        # plt.savefig("mygraph.png")
        plt.show()

    def save_result(self, output_dir):
        """Save the optimizer information, including best solver info and tree visualization.

        Parameters
        ----------
        output_dir : pathlib.Path
            full path of the output directory
        """
        self.best_solver.save_tree(output_dir)
        self.best_solver.save_best_solution(output_dir)
        for i in range(self.best_solver.tube_num):
            self.best_solver.visualize_full_search(output_dir, tube_num=i, with_solution=True)

        filename = output_dir / "optimize_result.txt"
        # noinspection PyTypeChecker
        optimize_result = open(filename, "w")

        optimize_result.write(f"Optimizer Success: {self.success}\n")
        optimize_result.write(f"Goal Reached: {self.best_solver.found_solution}\n")
        optimize_result.write(f"Completion Time: {self.completion_time}\n")
        optimize_result.write(f"Number of Iterations: {self.num_iterations}\n")
        optimize_result.write(f"Number of Function Evals: {self.num_function_eval}\n")
        q = ", ".join(str(i) for i in self.best_q)
        optimize_result.write(f"Optimized Q: {q}\n")

        optimize_result.close()

        filename = output_dir / "optimize_process.txt"
        # noinspection PyTypeChecker
        optimize_process = open(filename, "w")

        for i in range(len(self.optimize_process)):
            q = ", ".join(str(j) for j in self.optimize_process[i].get('q'))
            cost = self.optimize_process[i].get('cost')
            optimize_process.write(f"{q} | {cost}\n")

        optimize_process.close()
